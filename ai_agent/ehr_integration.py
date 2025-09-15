# ai_agent/ehr_integration.py
"""
EHR integration module for seamless healthcare system connectivity.
Supports FHIR R4, HL7, and custom EHR APIs.
"""
import json
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib
import logging
from dataclasses import dataclass

@dataclass
class Patient:
    """Patient data model following FHIR R4 standard."""
    id: str
    name: str
    birth_date: str
    gender: str
    contact: Dict[str, str]
    medical_record_number: str
    active: bool = True

@dataclass
class Observation:
    """Medical observation following FHIR R4 standard."""
    id: str
    patient_id: str
    code: str
    display: str
    value: Any
    unit: str
    timestamp: str
    status: str = "final"

class FHIRClient:
    """FHIR R4 compliant client for EHR integration."""
    
    def __init__(self, base_url: str, auth_token: str = None):
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.session = requests.Session()
        if auth_token:
            self.session.headers.update({'Authorization': f'Bearer {auth_token}'})
        self.session.headers.update({'Content-Type': 'application/fhir+json'})
        
    def get_patient(self, patient_id: str) -> Optional[Patient]:
        """Retrieve patient data from EHR."""
        try:
            response = self.session.get(f"{self.base_url}/Patient/{patient_id}")
            response.raise_for_status()
            
            data = response.json()
            return Patient(
                id=data['id'],
                name=data['name'][0]['given'][0] + ' ' + data['name'][0]['family'],
                birth_date=data['birthDate'],
                gender=data['gender'],
                contact=self._extract_contact(data.get('telecom', [])),
                medical_record_number=data.get('identifier', [{}])[0].get('value', ''),
                active=data.get('active', True)
            )
        except Exception as e:
            logging.error(f"Error retrieving patient {patient_id}: {e}")
            return None
    
    def create_observation(self, observation: Observation) -> bool:
        """Create new observation in EHR."""
        fhir_observation = {
            "resourceType": "Observation",
            "id": observation.id,
            "status": observation.status,
            "code": {
                "coding": [{
                    "code": observation.code,
                    "display": observation.display
                }]
            },
            "subject": {
                "reference": f"Patient/{observation.patient_id}"
            },
            "valueQuantity": {
                "value": observation.value,
                "unit": observation.unit
            },
            "effectiveDateTime": observation.timestamp
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/Observation",
                json=fhir_observation
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logging.error(f"Error creating observation: {e}")
            return False
    
    def get_patient_observations(self, patient_id: str, 
                               code: str = None) -> List[Observation]:
        """Retrieve patient observations."""
        params = {'patient': patient_id}
        if code:
            params['code'] = code
            
        try:
            response = self.session.get(
                f"{self.base_url}/Observation",
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            observations = []
            
            for entry in data.get('entry', []):
                resource = entry['resource']
                obs = Observation(
                    id=resource['id'],
                    patient_id=patient_id,
                    code=resource['code']['coding'][0]['code'],
                    display=resource['code']['coding'][0]['display'],
                    value=resource.get('valueQuantity', {}).get('value'),
                    unit=resource.get('valueQuantity', {}).get('unit', ''),
                    timestamp=resource.get('effectiveDateTime', ''),
                    status=resource.get('status', 'final')
                )
                observations.append(obs)
            
            return observations
        except Exception as e:
            logging.error(f"Error retrieving observations: {e}")
            return []
    
    def _extract_contact(self, telecom: List[Dict]) -> Dict[str, str]:
        """Extract contact information from FHIR telecom."""
        contact = {}
        for item in telecom:
            if item.get('system') == 'phone':
                contact['phone'] = item.get('value', '')
            elif item.get('system') == 'email':
                contact['email'] = item.get('value', '')
        return contact

class EHRIntegrationManager:
    """Manages integration with multiple EHR systems."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fhir_client = None
        self.logger = logging.getLogger('EHRIntegration')
        
        if config.get('fhir_enabled'):
            self.fhir_client = FHIRClient(
                config['fhir_base_url'],
                config.get('fhir_auth_token')
            )
    
    def sync_patient_data(self, patient_id: str) -> Dict[str, Any]:
        """Synchronize patient data from EHR."""
        if not self.fhir_client:
            return {'error': 'FHIR client not configured'}
        
        patient = self.fhir_client.get_patient(patient_id)
        if not patient:
            return {'error': 'Patient not found'}
        
        # Get recent observations
        observations = self.fhir_client.get_patient_observations(patient_id)
        
        # Get vital signs
        vital_signs = self._extract_vital_signs(observations)
        
        # Get lab results
        lab_results = self._extract_lab_results(observations)
        
        return {
            'patient': patient,
            'vital_signs': vital_signs,
            'lab_results': lab_results,
            'last_sync': datetime.now().isoformat()
        }
    
    def submit_ai_assessment(self, patient_id: str, assessment: Dict[str, Any]) -> bool:
        """Submit AI assessment results to EHR."""
        if not self.fhir_client:
            return False
        
        # Create observation for AI diagnosis
        diagnosis_obs = Observation(
            id=self._generate_id(),
            patient_id=patient_id,
            code="AI-DIAGNOSIS",
            display="AI-Generated Diagnosis",
            value=assessment['primary_diagnosis'],
            unit="",
            timestamp=datetime.now().isoformat()
        )
        
        # Create observation for confidence score
        confidence_obs = Observation(
            id=self._generate_id(),
            patient_id=patient_id,
            code="AI-CONFIDENCE",
            display="AI Diagnosis Confidence",
            value=assessment['confidence'],
            unit="%",
            timestamp=datetime.now().isoformat()
        )
        
        # Submit observations
        success1 = self.fhir_client.create_observation(diagnosis_obs)
        success2 = self.fhir_client.create_observation(confidence_obs)
        
        return success1 and success2
    
    def _extract_vital_signs(self, observations: List[Observation]) -> Dict[str, float]:
        """Extract vital signs from observations."""
        vital_codes = {
            '8480-6': 'systolic_bp',
            '8462-4': 'diastolic_bp',
            '8867-4': 'heart_rate',
            '9279-1': 'respiratory_rate',
            '8310-5': 'temperature',
            '2708-6': 'oxygen_saturation'
        }
        
        vitals = {}
        for obs in observations:
            if obs.code in vital_codes and obs.value is not None:
                vitals[vital_codes[obs.code]] = float(obs.value)
        
        return vitals
    
    def _extract_lab_results(self, observations: List[Observation]) -> Dict[str, Any]:
        """Extract lab results from observations."""
        lab_codes = {
            '33747-0': 'hemoglobin',
            '6690-2': 'white_blood_cells',
            '777-3': 'platelets',
            '2951-2': 'sodium',
            '2823-3': 'potassium',
            '2160-0': 'creatinine'
        }
        
        labs = {}
        for obs in observations:
            if obs.code in lab_codes and obs.value is not None:
                labs[lab_codes[obs.code]] = {
                    'value': obs.value,
                    'unit': obs.unit,
                    'timestamp': obs.timestamp
                }
        
        return labs
    
    def _generate_id(self) -> str:
        """Generate unique ID for observations."""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]

class HL7MessageHandler:
    """Handle HL7 v2.x messages for legacy system integration."""
    
    def __init__(self):
        self.field_separator = '|'
        self.component_separator = '^'
        self.repetition_separator = '~'
        self.escape_character = '\\'
        self.subcomponent_separator = '&'
    
    def parse_adt_message(self, hl7_message: str) -> Dict[str, Any]:
        """Parse ADT (Admit/Discharge/Transfer) message."""
        segments = hl7_message.strip().split('\r')
        patient_data = {}
        
        for segment in segments:
            fields = segment.split(self.field_separator)
            segment_type = fields[0]
            
            if segment_type == 'PID':  # Patient Identification
                patient_data.update(self._parse_pid_segment(fields))
            elif segment_type == 'PV1':  # Patient Visit
                patient_data.update(self._parse_pv1_segment(fields))
        
        return patient_data
    
    def _parse_pid_segment(self, fields: List[str]) -> Dict[str, Any]:
        """Parse PID segment for patient demographics."""
        return {
            'patient_id': fields[3] if len(fields) > 3 else '',
            'name': fields[5] if len(fields) > 5 else '',
            'birth_date': fields[7] if len(fields) > 7 else '',
            'gender': fields[8] if len(fields) > 8 else ''
        }
    
    def _parse_pv1_segment(self, fields: List[str]) -> Dict[str, Any]:
        """Parse PV1 segment for visit information."""
        return {
            'visit_id': fields[1] if len(fields) > 1 else '',
            'patient_class': fields[2] if len(fields) > 2 else '',
            'assigned_location': fields[3] if len(fields) > 3 else '',
            'attending_doctor': fields[7] if len(fields) > 7 else ''
        }
    
    def create_oru_message(self, patient_id: str, results: Dict[str, Any]) -> str:
        """Create ORU (Observation Result) message for AI results."""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        
        msh = f"MSH{self.field_separator}^~\\&{self.field_separator}AI_SYSTEM{self.field_separator}HOSPITAL{self.field_separator}{timestamp}{self.field_separator}{self.field_separator}ORU^R01{self.field_separator}{timestamp}{self.field_separator}P{self.field_separator}2.5"
        
        pid = f"PID{self.field_separator}1{self.field_separator}{self.field_separator}{patient_id}"
        
        obx_segments = []
        for i, (key, value) in enumerate(results.items(), 1):
            obx = f"OBX{self.field_separator}{i}{self.field_separator}ST{self.field_separator}{key}{self.field_separator}{self.field_separator}{value}{self.field_separator}{self.field_separator}{self.field_separator}F"
            obx_segments.append(obx)
        
        return '\r'.join([msh, pid] + obx_segments)