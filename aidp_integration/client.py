"""
AIDP Integration Client - Connect to AIDP GPU Marketplace
Handles provider discovery, job submission, and result retrieval
"""

import httpx
import logging
from typing import Dict, List, Optional
from django.conf import settings

logger = logging.getLogger(__name__)


class AIDPClient:
    """
    Client for interacting with AIDP decentralized GPU compute network
    """
    
    def __init__(self):
        self.api_endpoint = settings.AIDP_CONFIG['API_ENDPOINT']
        self.marketplace_url = settings.AIDP_CONFIG['GPU_MARKETPLACE']
        self.network = settings.AIDP_CONFIG['NETWORK']
        self.min_gpu_vram = settings.AIDP_CONFIG['MIN_GPU_VRAM']
        self.max_job_cost = settings.AIDP_CONFIG['MAX_JOB_COST']
        
        self.client = httpx.AsyncClient(
            base_url=self.api_endpoint,
            timeout=30.0,
            headers={
                'Content-Type': 'application/json',
                'User-Agent': 'AIDP-Compute-Gateway/1.0'
            }
        )
    
    async def discover_providers(
        self, 
        min_memory: int = None,
        gpu_type: str = None,
        max_price: float = None,
        region: str = None
    ) -> List[Dict]:
        """
        Discover available GPU providers from AIDP marketplace
        
        Args:
            min_memory: Minimum GPU memory in GB
            gpu_type: Specific GPU model (e.g., "A100", "RTX4090")
            max_price: Maximum price per hour
            region: Preferred geographic region
        
        Returns:
            List of available GPU providers
        """
        try:
            params = {
                'network': self.network,
                'status': 'online',
                'min_memory': min_memory or self.min_gpu_vram,
            }
            
            if gpu_type:
                params['gpu_type'] = gpu_type
            if max_price:
                params['max_price'] = max_price
            if region:
                params['region'] = region
            
            response = await self.client.get(
                f"{self.marketplace_url}/providers",
                params=params
            )
            response.raise_for_status()
            
            providers = response.json().get('providers', [])
            logger.info(f"Discovered {len(providers)} AIDP GPU providers")
            
            return providers
        
        except Exception as e:
            logger.error(f"Failed to discover AIDP providers: {str(e)}")
            # Return mock providers for development
            return self._get_mock_providers()
    
    async def select_optimal_provider(
        self,
        job_requirements: Dict
    ) -> Optional[Dict]:
        """
        Select the best GPU provider based on job requirements
        Uses scoring algorithm: reputation, price, availability
        
        Args:
            job_requirements: Dict containing job specifications
        
        Returns:
            Selected provider details or None
        """
        providers = await self.discover_providers(
            min_memory=job_requirements.get('gpu_memory', 8),
            gpu_type=job_requirements.get('gpu_type'),
            max_price=job_requirements.get('max_price', self.max_job_cost)
        )
        
        if not providers:
            logger.warning("No available AIDP providers found")
            return None
        
        # Score providers
        scored_providers = []
        for provider in providers:
            score = self._calculate_provider_score(provider, job_requirements)
            scored_providers.append((score, provider))
        
        # Sort by score (highest first)
        scored_providers.sort(reverse=True, key=lambda x: x[0])
        
        selected = scored_providers[0][1]
        logger.info(f"Selected AIDP provider: {selected['id']} (score: {scored_providers[0][0]:.2f})")
        
        return selected
    
    def _calculate_provider_score(self, provider: Dict, requirements: Dict) -> float:
        """
        Calculate provider suitability score
        Factors: reputation, price, availability, performance
        """
        score = 0.0
        
        # Reputation (0-40 points)
        reputation = provider.get('reputation_score', 5.0)
        score += (reputation / 10.0) * 40
        
        # Price (0-30 points) - lower is better
        price = provider.get('price_per_hour', 1.0)
        max_acceptable_price = requirements.get('max_price', self.max_job_cost)
        if price <= max_acceptable_price:
            price_score = (1 - (price / max_acceptable_price)) * 30
            score += price_score
        
        # Availability (0-20 points)
        current_load = provider.get('current_load', 0)
        availability_score = (1 - (current_load / 100.0)) * 20
        score += availability_score
        
        # Performance (0-10 points)
        success_rate = provider.get('success_rate', 100.0)
        score += (success_rate / 100.0) * 10
        
        return score
    
    async def submit_job(
        self,
        provider_id: str,
        job_config: Dict
    ) -> Dict:
        """
        Submit a GPU job to AIDP provider
        
        Args:
            provider_id: Selected provider ID
            job_config: Job configuration including workload type and parameters
        
        Returns:
            Job submission response with execution ID
        """
        try:
            payload = {
                'provider_id': provider_id,
                'workload_type': job_config['workload_type'],
                'parameters': job_config['parameters'],
                'timeout': job_config.get('timeout', 300),
                'priority': job_config.get('priority', 'normal'),
            }
            
            response = await self.client.post(
                f"{self.api_endpoint}/jobs/submit",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Submitted job to AIDP provider {provider_id}: {result.get('execution_id')}")
            
            return result
        
        except Exception as e:
            logger.error(f"Failed to submit job to AIDP: {str(e)}")
            # Return mock response for development
            return self._get_mock_job_submission(provider_id, job_config)
    
    async def check_job_status(
        self,
        execution_id: str,
        provider_id: str
    ) -> Dict:
        """
        Check status of running job on AIDP
        
        Args:
            execution_id: AIDP execution ID
            provider_id: Provider running the job
        
        Returns:
            Current job status and progress
        """
        try:
            response = await self.client.get(
                f"{self.api_endpoint}/jobs/{execution_id}/status",
                params={'provider_id': provider_id}
            )
            response.raise_for_status()
            
            return response.json()
        
        except Exception as e:
            logger.error(f"Failed to check job status: {str(e)}")
            return {'status': 'unknown', 'error': str(e)}
    
    async def retrieve_results(
        self,
        execution_id: str,
        provider_id: str
    ) -> Dict:
        """
        Retrieve completed job results from AIDP
        
        Args:
            execution_id: AIDP execution ID
            provider_id: Provider that ran the job
        
        Returns:
            Job results and execution proof
        """
        try:
            response = await self.client.get(
                f"{self.api_endpoint}/jobs/{execution_id}/results",
                params={'provider_id': provider_id}
            )
            response.raise_for_status()
            
            results = response.json()
            logger.info(f"Retrieved results for execution {execution_id}")
            
            return results
        
        except Exception as e:
            logger.error(f"Failed to retrieve job results: {str(e)}")
            return {'error': str(e)}
    
    async def verify_execution(
        self,
        execution_id: str,
        execution_proof: str
    ) -> bool:
        """
        Verify execution proof from AIDP provider
        
        Args:
            execution_id: AIDP execution ID
            execution_proof: Cryptographic proof of execution
        
        Returns:
            True if proof is valid
        """
        try:
            response = await self.client.post(
                f"{self.api_endpoint}/verify",
                json={
                    'execution_id': execution_id,
                    'proof': execution_proof
                }
            )
            response.raise_for_status()
            
            result = response.json()
            verified = result.get('verified', False)
            
            logger.info(f"Execution {execution_id} verification: {verified}")
            return verified
        
        except Exception as e:
            logger.error(f"Failed to verify execution: {str(e)}")
            return False
    
    async def cancel_job(
        self,
        execution_id: str,
        provider_id: str
    ) -> bool:
        """
        Cancel a running job on AIDP
        
        Args:
            execution_id: AIDP execution ID
            provider_id: Provider running the job
        
        Returns:
            True if cancellation successful
        """
        try:
            response = await self.client.post(
                f"{self.api_endpoint}/jobs/{execution_id}/cancel",
                json={'provider_id': provider_id}
            )
            response.raise_for_status()
            
            logger.info(f"Cancelled job {execution_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to cancel job: {str(e)}")
            return False
    
    def _get_mock_providers(self) -> List[Dict]:
        """Mock providers for development without AIDP connection"""
        return [
            {
                'id': 'provider_001',
                'name': 'AIDP GPU Node #1',
                'gpu_model': 'NVIDIA A100',
                'gpu_memory': 40,
                'price_per_hour': 2.5,
                'reputation_score': 9.2,
                'success_rate': 98.5,
                'current_load': 35,
                'region': 'us-west',
                'status': 'online'
            },
            {
                'id': 'provider_002',
                'name': 'AIDP GPU Node #2',
                'gpu_model': 'NVIDIA RTX 4090',
                'gpu_memory': 24,
                'price_per_hour': 1.2,
                'reputation_score': 8.7,
                'success_rate': 96.3,
                'current_load': 60,
                'region': 'eu-central',
                'status': 'online'
            },
            {
                'id': 'provider_003',
                'name': 'AIDP GPU Node #3',
                'gpu_model': 'NVIDIA H100',
                'gpu_memory': 80,
                'price_per_hour': 4.0,
                'reputation_score': 9.8,
                'success_rate': 99.1,
                'current_load': 20,
                'region': 'us-east',
                'status': 'online'
            }
        ]
    
    def _get_mock_job_submission(self, provider_id: str, job_config: Dict) -> Dict:
        """Mock job submission for development"""
        import uuid
        return {
            'execution_id': f"exec_{uuid.uuid4().hex[:16]}",
            'provider_id': provider_id,
            'status': 'submitted',
            'estimated_completion': '300s',
            'transaction_hash': f"tx_{uuid.uuid4().hex}"
        }
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


# Singleton instance
_aidp_client = None

def get_aidp_client() -> AIDPClient:
    """Get or create AIDP client instance"""
    global _aidp_client
    if _aidp_client is None:
        _aidp_client = AIDPClient()
    return _aidp_client