"""
GPU Jobs Models - Core job tracking and execution
"""

import uuid
from django.db import models
from django.conf import settings
from django.utils import timezone


class GPUJob(models.Model):
    """
    Main GPU job model tracking entire lifecycle
    """
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('queued', 'Queued'),
        ('routing', 'Routing to GPU'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]
    
    WORKLOAD_TYPES = [
        ('image_generation', 'Image Generation (Stable Diffusion)'),
        ('llm_inference', 'LLM Inference'),
        ('video_rendering', 'Video Rendering'),
        ('matrix_computation', 'Matrix Computation'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='gpu_jobs')
    
    # Job configuration
    name = models.CharField(max_length=255)
    workload_type = models.CharField(max_length=50, choices=WORKLOAD_TYPES)
    parameters = models.JSONField(default=dict)
    
    # Status tracking
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending', db_index=True)
    progress_percentage = models.IntegerField(default=0)
    
    # GPU requirements
    required_gpu_memory = models.IntegerField(default=8)  # GB
    required_gpu_type = models.CharField(max_length=100, blank=True)
    estimated_duration = models.IntegerField(null=True, blank=True)  # seconds
    
    # AIDP integration
    aidp_provider_id = models.CharField(max_length=255, blank=True, db_index=True)
    aidp_node_id = models.CharField(max_length=255, blank=True)
    aidp_transaction_hash = models.CharField(max_length=128, blank=True)
    
    # Execution tracking
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    duration_seconds = models.IntegerField(null=True, blank=True)
    
    # Cost tracking
    estimated_cost = models.DecimalField(max_digits=20, decimal_places=8, default=0)
    actual_cost = models.DecimalField(max_digits=20, decimal_places=8, default=0)
    
    # Results
    result_file = models.FileField(upload_to='results/', null=True, blank=True)
    result_data = models.JSONField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    
    # Verification
    execution_proof = models.TextField(blank=True)
    verified = models.BooleanField(default=False)
    
    # Metadata
    priority = models.IntegerField(default=0)
    retry_count = models.IntegerField(default=0)
    max_retries = models.IntegerField(default=3)
    
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'gpu_jobs'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status', 'created_at']),
            models.Index(fields=['user', 'status']),
            models.Index(fields=['aidp_provider_id']),
        ]
    
    def __str__(self):
        return f"{self.name} - {self.status}"
    
    def update_status(self, new_status, error_message=''):
        """Update job status with timestamp tracking"""
        self.status = new_status
        self.updated_at = timezone.now()
        
        if new_status == 'running' and not self.started_at:
            self.started_at = timezone.now()
        
        elif new_status in ['completed', 'failed', 'cancelled']:
            if not self.completed_at:
                self.completed_at = timezone.now()
            
            if self.started_at:
                delta = self.completed_at - self.started_at
                self.duration_seconds = int(delta.total_seconds())
        
        if error_message:
            self.error_message = error_message
        
        self.save()
    
    def can_retry(self):
        """Check if job can be retried"""
        return self.retry_count < self.max_retries and self.status == 'failed'


class GPUJobLog(models.Model):
    """
    Detailed logs for GPU job execution
    """
    
    LOG_LEVELS = [
        ('debug', 'Debug'),
        ('info', 'Info'),
        ('warning', 'Warning'),
        ('error', 'Error'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.ForeignKey(GPUJob, on_delete=models.CASCADE, related_name='logs')
    
    level = models.CharField(max_length=20, choices=LOG_LEVELS, default='info')
    message = models.TextField()
    metadata = models.JSONField(default=dict, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    
    class Meta:
        db_table = 'gpu_job_logs'
        ordering = ['created_at']
        indexes = [
            models.Index(fields=['job', 'created_at']),
        ]
    
    def __str__(self):
        return f"{self.job.name} - {self.level}: {self.message[:50]}"


class GPUProvider(models.Model):
    """
    Track AIDP GPU providers and their status
    """
    
    STATUS_CHOICES = [
        ('online', 'Online'),
        ('offline', 'Offline'),
        ('maintenance', 'Maintenance'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Provider identification
    aidp_provider_id = models.CharField(max_length=255, unique=True, db_index=True)
    name = models.CharField(max_length=255)
    wallet_address = models.CharField(max_length=44, blank=True)
    
    # GPU specifications
    gpu_model = models.CharField(max_length=100)
    gpu_memory = models.IntegerField()  # GB
    gpu_count = models.IntegerField(default=1)
    compute_capability = models.CharField(max_length=50, blank=True)
    
    # Status and availability
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='online')
    is_available = models.BooleanField(default=True)
    current_load = models.IntegerField(default=0)  # percentage
    
    # Pricing
    price_per_hour = models.DecimalField(max_digits=20, decimal_places=8)
    price_per_job = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    
    # Performance metrics
    average_response_time = models.FloatField(default=0)  # seconds
    success_rate = models.FloatField(default=100.0)  # percentage
    total_jobs_completed = models.IntegerField(default=0)
    
    # Reputation
    reputation_score = models.FloatField(default=5.0)  # 0-10 scale
    
    # Geographic data
    region = models.CharField(max_length=100, blank=True)
    country = models.CharField(max_length=100, blank=True)
    
    # Timestamps
    last_seen_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'gpu_providers'
        ordering = ['-reputation_score', '-success_rate']
    
    def __str__(self):
        return f"{self.name} ({self.gpu_model})"
    
    def update_metrics(self, job_success=True, response_time=None):
        """Update provider performance metrics"""
        self.total_jobs_completed += 1
        
        # Update success rate
        success_count = self.total_jobs_completed * (self.success_rate / 100)
        if job_success:
            success_count += 1
        self.success_rate = (success_count / self.total_jobs_completed) * 100
        
        # Update average response time
        if response_time:
            total_time = self.average_response_time * (self.total_jobs_completed - 1)
            self.average_response_time = (total_time + response_time) / self.total_jobs_completed
        
        self.save()


class GPUJobExecution(models.Model):
    """
    Track individual execution attempts on AIDP nodes
    """
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.ForeignKey(GPUJob, on_delete=models.CASCADE, related_name='executions')
    provider = models.ForeignKey(GPUProvider, on_delete=models.SET_NULL, null=True, related_name='executions')
    
    # Execution details
    attempt_number = models.IntegerField(default=1)
    node_assignment = models.CharField(max_length=255, blank=True)
    
    # AIDP tracking
    aidp_execution_id = models.CharField(max_length=255, blank=True)
    aidp_transaction_hash = models.CharField(max_length=128, blank=True)
    
    # Status
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    success = models.BooleanField(default=False)
    
    # Performance
    execution_time_seconds = models.IntegerField(null=True, blank=True)
    gpu_utilization = models.FloatField(null=True, blank=True)  # percentage
    memory_used = models.FloatField(null=True, blank=True)  # GB
    
    # Cost
    cost = models.DecimalField(max_digits=20, decimal_places=8, default=0)
    
    # Verification
    execution_proof = models.TextField(blank=True)
    proof_verified = models.BooleanField(default=False)
    
    # Error handling
    error_message = models.TextField(blank=True)
    error_code = models.CharField(max_length=50, blank=True)
    
    class Meta:
        db_table = 'gpu_job_executions'
        ordering = ['-started_at']
        indexes = [
            models.Index(fields=['job', 'attempt_number']),
        ]
    
    def __str__(self):
        return f"Execution #{self.attempt_number} for {self.job.name}"