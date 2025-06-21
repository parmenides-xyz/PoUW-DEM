"""
SGX Interface Module for Secure Grid Task Execution
Provides secure enclave functionality for proof generation
"""

import hashlib
import json
import time
import secrets
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend


class AttestationStatus(Enum):
    """SGX attestation status"""
    NOT_INITIALIZED = 0
    INITIALIZED = 1
    ATTESTED = 2
    FAILED = 3


@dataclass
class SGXQuote:
    """SGX attestation quote structure"""
    mrenclave: bytes  # Measurement of enclave code
    mrsigner: bytes   # Measurement of enclave signer
    isv_prod_id: int  # Product ID
    isv_svn: int      # Security version number
    report_data: bytes # Custom data in quote
    signature: bytes   # Quote signature


@dataclass
class TaskExecutionProof:
    """Proof of task execution in SGX"""
    task_id: int
    task_type: str
    input_hash: bytes
    output_hash: bytes
    computation_time: float
    energy_used: float
    grid_impact: float
    mrenclave: bytes
    timestamp: int
    signature: bytes


class SGXInterface:
    """
    Interface for SGX secure enclave operations
    In production, this would interface with actual SGX hardware
    For demo, we simulate SGX functionality
    """
    
    def __init__(self):
        self.status = AttestationStatus.NOT_INITIALIZED
        self.enclave_id = None
        self.attestation_key = None
        self.signing_key = None
        self.mrenclave = None
        self.mrsigner = None
        
        # Simulated secure storage
        self._secure_storage = {}
        
        # Initialize keys
        self._initialize_keys()
        
    def _initialize_keys(self):
        """Initialize cryptographic keys for the enclave"""
        # Generate RSA key pair for attestation
        self.attestation_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        # Generate signing key
        self.signing_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        # Simulate enclave measurements
        self.mrenclave = secrets.token_bytes(32)
        self.mrsigner = secrets.token_bytes(32)
        
        self.status = AttestationStatus.INITIALIZED
    
    def create_enclave(self, enclave_path: str = "grid_optimization.signed.so") -> bool:
        """
        Create and initialize SGX enclave
        In production, this would load the actual enclave binary
        """
        try:
            # Simulate enclave creation
            self.enclave_id = secrets.token_hex(16)
            
            # Initialize enclave with grid optimization code
            self._secure_storage['enclave_code'] = self._load_enclave_code(enclave_path)
            
            self.status = AttestationStatus.ATTESTED
            return True
            
        except Exception as e:
            print(f"Failed to create enclave: {e}")
            self.status = AttestationStatus.FAILED
            return False
    
    def _load_enclave_code(self, enclave_path: str) -> Dict:
        """Load enclave code (simulated)"""
        return {
            'stability_simulation': self._stability_simulation_code,
            'renewable_forecast': self._renewable_forecast_code,
            'load_balance': self._load_balance_code,
            'frequency_regulation': self._frequency_regulation_code,
            'voltage_optimization': self._voltage_optimization_code
        }
    
    def get_remote_attestation_quote(self, user_data: bytes = b"") -> SGXQuote:
        """
        Generate remote attestation quote
        This proves the enclave is running genuine code
        """
        if self.status != AttestationStatus.ATTESTED:
            raise RuntimeError("Enclave not properly initialized")
        
        # Include user data in quote
        report_data = hashlib.sha256(user_data).digest()
        
        # Create quote structure
        quote_data = self.mrenclave + self.mrsigner + report_data
        
        # Sign quote with attestation key
        signature = self.attestation_key.sign(
            quote_data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return SGXQuote(
            mrenclave=self.mrenclave,
            mrsigner=self.mrsigner,
            isv_prod_id=1,
            isv_svn=1,
            report_data=report_data,
            signature=signature
        )
    
    def execute_grid_task(self, 
                         task_type: str,
                         task_data: Dict,
                         task_id: int) -> TaskExecutionProof:
        """
        Execute grid optimization task in secure enclave
        """
        if self.status != AttestationStatus.ATTESTED:
            raise RuntimeError("Enclave not attested")
        
        start_time = time.time()
        
        # Hash input data
        input_bytes = json.dumps(task_data, sort_keys=True).encode()
        input_hash = hashlib.sha256(input_bytes).digest()
        
        # Execute task in "secure" environment
        if task_type == 'stability_simulation':
            result = self._stability_simulation_code(task_data)
        elif task_type == 'renewable_forecast':
            result = self._renewable_forecast_code(task_data)
        elif task_type == 'load_balance':
            result = self._load_balance_code(task_data)
        elif task_type == 'frequency_regulation':
            result = self._frequency_regulation_code(task_data)
        elif task_type == 'voltage_optimization':
            result = self._voltage_optimization_code(task_data)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        computation_time = time.time() - start_time
        
        # Hash output
        output_bytes = json.dumps(result, sort_keys=True).encode()
        output_hash = hashlib.sha256(output_bytes).digest()
        
        # Create proof
        proof_data = (
            task_id.to_bytes(8, 'big') +
            task_type.encode() +
            input_hash +
            output_hash +
            int(computation_time * 1000).to_bytes(8, 'big') +
            int(result['energy_used'] * 1000).to_bytes(8, 'big') +
            int(result['grid_impact'] * 1000).to_bytes(8, 'big') +
            self.mrenclave +
            int(time.time()).to_bytes(8, 'big')
        )
        
        # Sign proof
        signature = self.signing_key.sign(
            proof_data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return TaskExecutionProof(
            task_id=task_id,
            task_type=task_type,
            input_hash=input_hash,
            output_hash=output_hash,
            computation_time=computation_time,
            energy_used=result['energy_used'],
            grid_impact=result['grid_impact'],
            mrenclave=self.mrenclave,
            timestamp=int(time.time()),
            signature=signature
        )
    
    def _stability_simulation_code(self, data: Dict) -> Dict:
        """
        Secure implementation of grid stability simulation
        This runs inside the SGX enclave
        """
        # Extract grid parameters
        voltage = data.get('voltage', 230)
        frequency = data.get('frequency', 50)
        load = data.get('load', 0.8)
        generation = data.get('generation', 0.85)
        
        # Simulate grid dynamics
        imbalance = abs(generation - load)
        voltage_deviation = abs(voltage - 230) / 230
        frequency_deviation = abs(frequency - 50) / 50
        
        # Calculate stability metrics
        stability_index = 1.0 - (imbalance + voltage_deviation + frequency_deviation) / 3
        
        # Optimization recommendations
        if imbalance > 0.1:
            recommendation = "increase_storage_discharge" if generation < load else "increase_storage_charge"
        else:
            recommendation = "maintain_current_state"
        
        # Calculate grid impact (improvement percentage)
        grid_impact = max(0, min(5, imbalance * 10))  # 0-5% improvement
        
        return {
            'stability_index': stability_index,
            'recommendation': recommendation,
            'grid_impact': grid_impact,
            'energy_used': 0.05  # kWh for computation
        }
    
    def _renewable_forecast_code(self, data: Dict) -> Dict:
        """
        Secure implementation of renewable energy forecasting
        """
        # Historical data
        historical_production = data.get('historical_production', [])
        weather_data = data.get('weather_data', {})
        
        # Simple forecasting model (in production would use ML)
        if historical_production:
            avg_production = np.mean(historical_production)
            
            # Weather adjustments
            cloud_cover = weather_data.get('cloud_cover', 0.5)
            wind_speed = weather_data.get('wind_speed', 10)
            
            solar_factor = 1.0 - cloud_cover
            wind_factor = min(1.0, wind_speed / 15)
            
            forecast = avg_production * solar_factor * 0.7 + avg_production * wind_factor * 0.3
        else:
            forecast = 0
        
        # Calculate accuracy impact on grid planning
        forecast_accuracy = 0.85 + np.random.uniform(-0.05, 0.05)
        grid_impact = (forecast_accuracy - 0.8) * 20  # Convert to percentage
        
        return {
            'forecast': forecast,
            'confidence': forecast_accuracy,
            'grid_impact': max(0, grid_impact),
            'energy_used': 0.03
        }
    
    def _load_balance_code(self, data: Dict) -> Dict:
        """
        Secure implementation of load balancing optimization
        """
        loads = data.get('loads', [])
        capacity = data.get('capacity', [])
        
        if not loads or not capacity:
            return {'grid_impact': 0, 'energy_used': 0.02}
        
        # Calculate current imbalance
        total_load = sum(loads)
        total_capacity = sum(capacity)
        
        if total_capacity > 0:
            utilization = total_load / total_capacity
            imbalance = np.std(loads) / np.mean(loads) if np.mean(loads) > 0 else 0
            
            # Optimization impact
            potential_improvement = imbalance * 0.3  # 30% of imbalance can be improved
            grid_impact = min(3.0, potential_improvement * 100)
        else:
            grid_impact = 0
        
        return {
            'optimized_distribution': loads,  # Simplified
            'grid_impact': grid_impact,
            'energy_used': 0.04
        }
    
    def _frequency_regulation_code(self, data: Dict) -> Dict:
        """
        Secure implementation of frequency regulation
        """
        current_frequency = data.get('frequency', 50)
        target_frequency = 50  # Hz
        
        deviation = abs(current_frequency - target_frequency)
        
        # Calculate regulation response
        if deviation > 0.2:  # Critical deviation
            response = "fast_response_required"
            grid_impact = 4.0
        elif deviation > 0.1:
            response = "normal_response"
            grid_impact = 2.0
        else:
            response = "monitoring"
            grid_impact = 0.5
        
        return {
            'response': response,
            'grid_impact': grid_impact,
            'energy_used': 0.025
        }
    
    def _voltage_optimization_code(self, data: Dict) -> Dict:
        """
        Secure implementation of voltage optimization
        """
        voltage_readings = data.get('voltage_readings', [])
        
        if not voltage_readings:
            return {'grid_impact': 0, 'energy_used': 0.02}
        
        avg_voltage = np.mean(voltage_readings)
        voltage_variance = np.var(voltage_readings)
        
        # Target is 230V with minimal variance
        deviation_from_target = abs(avg_voltage - 230) / 230
        
        # Calculate optimization impact
        if deviation_from_target > 0.05:  # More than 5% deviation
            grid_impact = min(3.0, deviation_from_target * 30)
        else:
            grid_impact = voltage_variance * 10  # Reduce variance
        
        return {
            'optimized_voltage': 230,
            'grid_impact': grid_impact,
            'energy_used': 0.03
        }
    
    def verify_proof(self, proof: TaskExecutionProof) -> bool:
        """
        Verify a task execution proof
        """
        # Reconstruct proof data
        proof_data = (
            proof.task_id.to_bytes(8, 'big') +
            proof.task_type.encode() +
            proof.input_hash +
            proof.output_hash +
            int(proof.computation_time * 1000).to_bytes(8, 'big') +
            int(proof.energy_used * 1000).to_bytes(8, 'big') +
            int(proof.grid_impact * 1000).to_bytes(8, 'big') +
            proof.mrenclave +
            proof.timestamp.to_bytes(8, 'big')
        )
        
        try:
            # Verify signature
            public_key = self.signing_key.public_key()
            public_key.verify(
                proof.signature,
                proof_data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            # Verify enclave measurement
            if proof.mrenclave != self.mrenclave:
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_public_attestation_key(self) -> bytes:
        """Get public key for attestation verification"""
        return self.attestation_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    
    def destroy_enclave(self):
        """Destroy the enclave and clean up resources"""
        self._secure_storage.clear()
        self.enclave_id = None
        self.status = AttestationStatus.NOT_INITIALIZED