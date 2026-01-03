"""
GPU Jobs API Views
RESTful endpoints for job management
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter, OrderingFilter

from gpu_jobs.models import GPUJob, GPUProvider, GPUJobLog
from gpu_jobs.serializers import (
    GPUJobSerializer, GPUJobCreateSerializer,
    GPUProviderSerializer, GPUJobLogSerializer
)
from gpu_jobs.tasks import process_gpu_job


class GPUJobViewSet(viewsets.ModelViewSet):
    """
    API endpoints for GPU job management
    
    list: Get all jobs for authenticated user
    create: Submit new GPU job
    retrieve: Get specific job details
    update: Update job (limited fields)
    destroy: Delete job (only if not running)
    """
    
    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ['status', 'workload_type']
    search_fields = ['name', 'aidp_provider_id']
    ordering_fields = ['created_at', 'started_at', 'priority']
    ordering = ['-created_at']
    
    def get_queryset(self):
        """Return jobs for authenticated user only"""
        return GPUJob.objects.filter(user=self.request.user)
    
    def get_serializer_class(self):
        """Use different serializer for creation"""
        if self.action == 'create':
            return GPUJobCreateSerializer
        return GPUJobSerializer
    
    def perform_create(self, serializer):
        """Create job and trigger async processing"""
        job = serializer.save(user=self.request.user)
        
        # Trigger async job processing
        process_gpu_job.delay(str(job.id))
        
        return job
    
    @action(detail=True, methods=['get'])
    def results(self, request, pk=None):
        """
        Get job results
        
        GET /api/jobs/{id}/results/
        """
        job = self.get_object()
        
        if job.status != 'completed':
            return Response(
                {'error': 'Job not completed yet', 'status': job.status},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        result_data = {
            'job_id': str(job.id),
            'status': job.status,
            'result_data': job.result_data,
            'execution_proof': job.execution_proof,
            'verified': job.verified,
            'duration_seconds': job.duration_seconds,
            'actual_cost': str(job.actual_cost),
        }
        
        if job.result_file:
            result_data['result_file'] = request.build_absolute_uri(job.result_file.url)
        
        return Response(result_data)
    
    @action(detail=True, methods=['post'])
    def cancel(self, request, pk=None):
        """
        Cancel a running job
        
        POST /api/jobs/{id}/cancel/
        """
        job = self.get_object()
        
        if job.status not in ['pending', 'queued', 'routing', 'running']:
            return Response(
                {'error': f'Cannot cancel job with status: {job.status}'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        job.update_status('cancelled')
        
        return Response({
            'message': 'Job cancelled successfully',
            'status': job.status
        })
    
    @action(detail=True, methods=['post'])
    def retry(self, request, pk=None):
        """
        Retry a failed job
        
        POST /api/jobs/{id}/retry/
        """
        job = self.get_object()
        
        if not job.can_retry():
            return Response(
                {'error': 'Job cannot be retried'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        job.update_status('pending')
        job.retry_count += 1
        job.error_message = ''
        job.save()
        
        # Trigger async processing
        process_gpu_job.delay(str(job.id))
        
        return Response({
            'message': 'Job retry initiated',
            'retry_count': job.retry_count
        })
    
    @action(detail=True, methods=['get'])
    def logs(self, request, pk=None):
        """
        Get job execution logs
        
        GET /api/jobs/{id}/logs/
        """
        job = self.get_object()
        logs = job.logs.all()
        serializer = GPUJobLogSerializer(logs, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def stats(self, request):
        """
        Get user job statistics
        
        GET /api/jobs/stats/
        """
        user_jobs = self.get_queryset()
        
        stats = {
            'total_jobs': user_jobs.count(),
            'pending': user_jobs.filter(status='pending').count(),
            'running': user_jobs.filter(status='running').count(),
            'completed': user_jobs.filter(status='completed').count(),
            'failed': user_jobs.filter(status='failed').count(),
            'total_cost': str(sum(job.actual_cost for job in user_jobs)),
            'by_workload_type': {}
        }
        
        for workload_type, _ in GPUJob.WORKLOAD_TYPES:
            count = user_jobs.filter(workload_type=workload_type).count()
            if count > 0:
                stats['by_workload_type'][workload_type] = count
        
        return Response(stats)


class GPUProviderViewSet(viewsets.ReadOnlyModelViewSet):
    """
    API endpoints for GPU provider information
    
    list: Get all available AIDP GPU providers
    retrieve: Get specific provider details
    """
    
    queryset = GPUProvider.objects.filter(status='online')
    serializer_class = GPUProviderSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend, OrderingFilter]
    filterset_fields = ['gpu_model', 'status', 'region']
    ordering_fields = ['reputation_score', 'price_per_hour', 'success_rate']
    ordering = ['-reputation_score']
    
    @action(detail=True, methods=['get'])
    def performance(self, request, pk=None):
        """
        Get provider performance metrics
        
        GET /api/providers/{id}/performance/
        """
        provider = self.get_object()
        
        performance = {
            'provider_id': str(provider.id),
            'name': provider.name,
            'reputation_score': provider.reputation_score,
            'success_rate': provider.success_rate,
            'average_response_time': provider.average_response_time,
            'total_jobs_completed': provider.total_jobs_completed,
            'current_load': provider.current_load,
        }
        
        return Response(performance)


class HealthCheckView(viewsets.ViewSet):
    """
    System health check endpoint
    """
    
    permission_classes = []  # Public endpoint
    
    def list(self, request):
        """
        GET /api/health/
        """
        from django.db import connection
        from django.core.cache import cache
        
        health = {
            'status': 'healthy',
            'database': 'unknown',
            'redis': 'unknown',
            'celery': 'unknown'
        }
        
        # Check database
        try:
            connection.ensure_connection()
            health['database'] = 'connected'
        except Exception:
            health['database'] = 'disconnected'
            health['status'] = 'unhealthy'
        
        # Check Redis
        try:
            cache.set('health_check', 'ok', 10)
            if cache.get('health_check') == 'ok':
                health['redis'] = 'connected'
            else:
                health['redis'] = 'disconnected'
        except Exception:
            health['redis'] = 'disconnected'
            health['status'] = 'unhealthy'
        
        # Check Celery
        try:
            from aidp_gateway.celery import app
            inspect = app.control.inspect()
            active = inspect.active()
            if active:
                health['celery'] = 'active'
            else:
                health['celery'] = 'no_workers'
        except Exception:
            health['celery'] = 'unreachable'
        
        status_code = status.HTTP_200_OK if health['status'] == 'healthy' else status.HTTP_503_SERVICE_UNAVAILABLE
        return Response(health, status=status_code)