"""
Celery Tasks for GPU Job Processing
Handles async GPU job execution via AIDP
"""

import logging
from celery import shared_task
from django.core.files.base import ContentFile
from django.utils import timezone
from asgiref.sync import async_to_sync

from gpu_jobs.models import GPUJob, GPUJobLog, GPUJobExecution, GPUProvider
from aidp_integration.client import get_aidp_client
from aidp_integration.executor import get_gpu_executor

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3)
def process_gpu_job(self, job_id: str):
    """
    Main task to process GPU job through AIDP pipeline
    
    Pipeline:
    1. Select AIDP GPU provider
    2. Submit job to provider
    3. Monitor execution
    4. Retrieve and verify results
    """
    try:
        job = GPUJob.objects.get(id=job_id)
        job.update_status('queued')
        
        log_entry(job, 'info', f'Starting GPU job processing: {job.workload_type}')
        
        # Step 1: Select optimal AIDP provider
        job.update_status('routing')
        log_entry(job, 'info', 'Selecting AIDP GPU provider')
        
        provider = async_to_sync(select_aidp_provider)(job)
        if not provider:
            raise Exception("No suitable AIDP GPU provider available")
        
        job.aidp_provider_id = provider['id']
        job.aidp_node_id = provider.get('node_id', '')
        job.save()
        
        log_entry(job, 'info', f"Selected AIDP provider: {provider['name']} ({provider['gpu_model']})")
        
        # Step 2: Submit to AIDP
        job.update_status('running')
        log_entry(job, 'info', 'Submitting job to AIDP compute network')
        
        execution_result = async_to_sync(execute_on_aidp)(job, provider)
        
        # Step 3: Create execution record
        create_execution_record(job, provider, execution_result)
        
        # Step 4: Process results
        if execution_result.get('success'):
            process_job_results(job, execution_result)
            job.update_status('completed')
            log_entry(job, 'info', 'Job completed successfully')
        else:
            error_msg = execution_result.get('error', 'Unknown error')
            job.update_status('failed', error_msg)
            log_entry(job, 'error', f'Job failed: {error_msg}')
        
        # Update user metrics
        update_user_metrics(job)
        
        return {
            'job_id': str(job.id),
            'status': job.status,
            'execution_time': job.duration_seconds
        }
    
    except Exception as exc:
        logger.error(f"GPU job {job_id} processing failed: {str(exc)}", exc_info=True)
        
        try:
            job = GPUJob.objects.get(id=job_id)
            
            if job.can_retry():
                job.retry_count += 1
                job.save()
                log_entry(job, 'warning', f'Retrying job (attempt {job.retry_count + 1})')
                raise self.retry(exc=exc, countdown=60 * job.retry_count)
            else:
                job.update_status('failed', str(exc))
                log_entry(job, 'error', f'Job failed after {job.retry_count} retries')
        except GPUJob.DoesNotExist:
            pass
        
        raise


async def select_aidp_provider(job: GPUJob) -> dict:
    """Select optimal AIDP provider for job"""
    client = get_aidp_client()
    
    job_requirements = {
        'gpu_memory': job.required_gpu_memory,
        'gpu_type': job.required_gpu_type,
        'workload_type': job.workload_type,
        'max_price': float(job.estimated_cost) if job.estimated_cost else None
    }
    
    provider = await client.select_optimal_provider(job_requirements)
    
    if provider:
        # Update or create GPUProvider record
        gpu_provider, created = GPUProvider.objects.update_or_create(
            aidp_provider_id=provider['id'],
            defaults={
                'name': provider.get('name', 'Unknown Provider'),
                'gpu_model': provider.get('gpu_model', 'Unknown'),
                'gpu_memory': provider.get('gpu_memory', 8),
                'price_per_hour': provider.get('price_per_hour', 0),
                'status': 'online',
                'reputation_score': provider.get('reputation_score', 5.0),
                'success_rate': provider.get('success_rate', 100.0),
            }
        )
    
    return provider


async def execute_on_aidp(job: GPUJob, provider: dict) -> dict:
    """Execute job on AIDP GPU provider"""
    client = get_aidp_client()
    executor = get_gpu_executor()
    
    # Submit job to AIDP
    job_config = {
        'workload_type': job.workload_type,
        'parameters': job.parameters,
        'timeout': 300,
        'priority': job.priority
    }
    
    submission = await client.submit_job(provider['id'], job_config)
    
    job.aidp_transaction_hash = submission.get('transaction_hash', '')
    job.save()
    
    # Execute actual GPU workload
    result = await executor.execute_workload(
        job.workload_type,
        job.parameters,
        provider
    )
    
    # Verify execution with AIDP
    if result.get('success') and submission.get('execution_id'):
        execution_proof = result.get('execution_proof', 'mock_proof')
        verified = await client.verify_execution(
            submission['execution_id'],
            execution_proof
        )
        result['verified'] = verified
    
    return result


def create_execution_record(job: GPUJob, provider: dict, execution_result: dict):
    """Create execution record for tracking"""
    try:
        gpu_provider = GPUProvider.objects.get(aidp_provider_id=provider['id'])
    except GPUProvider.DoesNotExist:
        gpu_provider = None
    
    execution = GPUJobExecution.objects.create(
        job=job,
        provider=gpu_provider,
        attempt_number=job.retry_count + 1,
        success=execution_result.get('success', False),
        execution_time_seconds=job.duration_seconds,
        cost=job.actual_cost,
        execution_proof=execution_result.get('execution_proof', ''),
        proof_verified=execution_result.get('verified', False),
        error_message=execution_result.get('error', '')
    )
    
    # Update provider metrics
    if gpu_provider:
        gpu_provider.update_metrics(
            job_success=execution_result.get('success', False),
            response_time=job.duration_seconds
        )


def process_job_results(job: GPUJob, execution_result: dict):
    """Process and save job results"""
    # Save result data
    job.result_data = execution_result.get('metadata', {})
    job.verified = execution_result.get('verified', False)
    job.execution_proof = execution_result.get('execution_proof', '')
    
    # Save result file if present
    if 'image_data' in execution_result:
        filename = f"{job.id}_result.png"
        job.result_file.save(
            filename,
            ContentFile(execution_result['image_data']),
            save=False
        )
    
    elif 'video_data' in execution_result:
        filename = f"{job.id}_result.mp4"
        job.result_file.save(
            filename,
            ContentFile(execution_result['video_data']),
            save=False
        )
    
    job.save()


def update_user_metrics(job: GPUJob):
    """Update user usage metrics"""
    user = job.user
    user.total_jobs_submitted += 1
    user.total_compute_cost += job.actual_cost
    user.save(update_fields=['total_jobs_submitted', 'total_compute_cost'])


def log_entry(job: GPUJob, level: str, message: str, metadata: dict = None):
    """Create a job log entry"""
    from gpu_jobs.models import GPUJobLog
    
    GPUJobLog.objects.create(
        job=job,
        level=level,
        message=message,
        metadata=metadata or {}
    )


@shared_task
def monitor_job_status(job_id: str):
    """
    Monitor job status on AIDP (periodic check)
    """
    try:
        job = GPUJob.objects.get(id=job_id)
        
        if job.status not in ['running', 'queued']:
            return
        
        client = get_aidp_client()
        
        # Check status with AIDP
        status_info = async_to_sync(client.check_job_status)(
            job.aidp_transaction_hash,
            job.aidp_provider_id
        )
        
        # Update progress
        if 'progress' in status_info:
            job.progress_percentage = status_info['progress']
            job.save()
        
        logger.info(f"Job {job_id} status: {status_info.get('status')}")
    
    except Exception as e:
        logger.error(f"Failed to monitor job {job_id}: {str(e)}")


@shared_task
def cleanup_old_jobs():
    """
    Cleanup old completed jobs (run daily)
    """
    from datetime import timedelta
    
    cutoff_date = timezone.now() - timedelta(days=30)
    
    old_jobs = GPUJob.objects.filter(
        status__in=['completed', 'failed', 'cancelled'],
        created_at__lt=cutoff_date
    )
    
    count = old_jobs.count()
    old_jobs.delete()
    
    logger.info(f"Cleaned up {count} old GPU jobs")
    return count


@shared_task
def sync_aidp_providers():
    """
    Sync GPU providers from AIDP marketplace (run hourly)
    """
    client = get_aidp_client()
    
    providers = async_to_sync(client.discover_providers)()
    
    synced_count = 0
    for provider_data in providers:
        GPUProvider.objects.update_or_create(
            aidp_provider_id=provider_data['id'],
            defaults={
                'name': provider_data.get('name', 'Unknown'),
                'gpu_model': provider_data.get('gpu_model', 'Unknown'),
                'gpu_memory': provider_data.get('gpu_memory', 8),
                'price_per_hour': provider_data.get('price_per_hour', 0),
                'status': provider_data.get('status', 'online'),
                'reputation_score': provider_data.get('reputation_score', 5.0),
                'success_rate': provider_data.get('success_rate', 100.0),
                'region': provider_data.get('region', ''),
            }
        )
        synced_count += 1
    
    logger.info(f"Synced {synced_count} AIDP GPU providers")
    return synced_count