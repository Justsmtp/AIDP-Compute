"""
Authentication Models - Wallet-based user system with API keys
"""

import uuid
import secrets
from django.contrib.auth.models.abstract_user import AbstractBaseUser, PermissionsMixin, BaseUserManager
from django.db import models
from django.utils import timezone


class UserManager(BaseUserManager):
    """Custom user manager for wallet-based authentication"""
    
    def create_user(self, wallet_address, **extra_fields):
        if not wallet_address:
            raise ValueError('Wallet address is required')
        
        user = self.model(wallet_address=wallet_address, **extra_fields)
        user.save(using=self._db)
        return user
    
    def create_superuser(self, wallet_address, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('is_active', True)
        
        return self.create_user(wallet_address, **extra_fields)


class User(AbstractBaseUser, PermissionsMixin):
    """
    Custom user model supporting Solana wallet authentication
    """
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    wallet_address = models.CharField(max_length=44, unique=True, db_index=True)
    username = models.CharField(max_length=150, blank=True, null=True)
    email = models.EmailField(blank=True, null=True)
    
    # Verification
    is_verified = models.BooleanField(default=False)
    nonce = models.CharField(max_length=64, blank=True, null=True)
    
    # Permissions
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    is_superuser = models.BooleanField(default=False)
    
    # Usage metrics
    total_jobs_submitted = models.IntegerField(default=0)
    total_compute_cost = models.DecimalField(max_digits=20, decimal_places=8, default=0)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_login = models.DateTimeField(null=True, blank=True)
    
    objects = UserManager()
    
    USERNAME_FIELD = 'wallet_address'
    REQUIRED_FIELDS = []
    
    class Meta:
        db_table = 'users'
        ordering = ['-created_at']
    
    def __str__(self):
        return self.wallet_address
    
    def generate_nonce(self):
        """Generate a random nonce for signature verification"""
        self.nonce = secrets.token_hex(32)
        self.save(update_fields=['nonce'])
        return self.nonce


class APIKey(models.Model):
    """
    API keys for programmatic access
    """
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='api_keys')
    
    name = models.CharField(max_length=255)
    key = models.CharField(max_length=64, unique=True, db_index=True)
    prefix = models.CharField(max_length=8)  # First 8 chars for identification
    
    # Permissions
    is_active = models.BooleanField(default=True)
    allowed_endpoints = models.JSONField(default=list, blank=True)
    
    # Rate limiting
    rate_limit_per_hour = models.IntegerField(default=1000)
    
    # Usage tracking
    last_used_at = models.DateTimeField(null=True, blank=True)
    total_requests = models.IntegerField(default=0)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        db_table = 'api_keys'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.prefix}...)"
    
    @staticmethod
    def generate_key():
        """Generate a secure API key"""
        key = f"aidp_{secrets.token_urlsafe(40)}"
        prefix = key[:12]
        return key, prefix
    
    def is_valid(self):
        """Check if API key is valid and not expired"""
        if not self.is_active:
            return False
        if self.expires_at and self.expires_at < timezone.now():
            return False
        return True
    
    def record_usage(self):
        """Record API key usage"""
        self.last_used_at = timezone.now()
        self.total_requests += 1
        self.save(update_fields=['last_used_at', 'total_requests'])


class WalletVerification(models.Model):
    """
    Track wallet verification attempts
    """
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    wallet_address = models.CharField(max_length=44, db_index=True)
    nonce = models.CharField(max_length=64)
    signature = models.TextField()
    
    verified = models.BooleanField(default=False)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'wallet_verifications'
        ordering = ['-created_at']