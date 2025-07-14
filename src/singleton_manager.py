"""
Singleton Instance Manager
=========================

This module provides singleton pattern implementation to ensure only one instance 
of the Quant AI Trader application runs at a time. This prevents conflicts, 
resource contention, and ensures data integrity.

Features:
- File-based locking mechanism
- Process ID tracking
- Graceful cleanup on exit
- Health monitoring of running instances
- Force cleanup for stale locks

Author: AI Assistant
Version: 1.0.0
License: MIT
"""

import os
import sys
import time
import signal
import psutil
import atexit
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SingletonManager:
    """
    Manages singleton application instances using file-based locking.
    
    This ensures only one instance of the application can run at a time,
    preventing conflicts and resource contention.
    """
    
    def __init__(self, app_name: str = "quant_ai_trader", lock_dir: str = "locks"):
        """
        Initialize the singleton manager.
        
        Args:
            app_name: Name of the application (used for lock file naming)
            lock_dir: Directory to store lock files
        """
        self.app_name = app_name
        self.lock_dir = Path(lock_dir)
        self.lock_file = self.lock_dir / f"{app_name}.lock"
        self.pid_file = self.lock_dir / f"{app_name}.pid"
        self.health_file = self.lock_dir / f"{app_name}.health"
        
        # Ensure lock directory exists
        self.lock_dir.mkdir(exist_ok=True)
        
        # Current process info
        self.current_pid = os.getpid()
        self.current_start_time = datetime.now()
        
        # Register cleanup handlers
        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        logger.info(f"Singleton manager initialized for {app_name}")
    
    def acquire_lock(self, force: bool = False) -> bool:
        """
        Acquire singleton lock for the application.
        
        Args:
            force: If True, force acquire lock even if another instance exists
            
        Returns:
            True if lock acquired successfully, False otherwise
        """
        try:
            # Check if lock already exists
            if self.lock_file.exists() and not force:
                if self._is_instance_running():
                    logger.error(f"Another instance of {self.app_name} is already running")
                    self._show_running_instance_info()
                    return False
                else:
                    logger.info("Found stale lock file, cleaning up...")
                    self._cleanup_stale_lock()
            
            # Create lock file
            with open(self.lock_file, 'w') as f:
                f.write(f"pid:{self.current_pid}\n")
                f.write(f"start_time:{self.current_start_time.isoformat()}\n")
                f.write(f"hostname:{os.uname().nodename}\n")
                f.write(f"user:{os.getenv('USER', 'unknown')}\n")
            
            # Create PID file
            with open(self.pid_file, 'w') as f:
                f.write(str(self.current_pid))
            
            # Create initial health file
            self._update_health_file()
            
            logger.info(f"Successfully acquired singleton lock for {self.app_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to acquire singleton lock: {str(e)}")
            return False
    
    def _is_instance_running(self) -> bool:
        """
        Check if another instance is currently running.
        
        Returns:
            True if another instance is running, False otherwise
        """
        try:
            if not self.pid_file.exists():
                return False
            
            # Read PID from file
            with open(self.pid_file, 'r') as f:
                pid_str = f.read().strip()
            
            if not pid_str.isdigit():
                return False
            
            pid = int(pid_str)
            
            # Check if process exists and is our application
            if psutil.pid_exists(pid):
                try:
                    process = psutil.Process(pid)
                    
                    # Check if it's our application (simplified check)
                    if self.app_name.lower() in ' '.join(process.cmdline()).lower():
                        # Check health file to ensure instance is responsive
                        return self._check_instance_health(pid)
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    return False
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking if instance is running: {str(e)}")
            return False
    
    def _check_instance_health(self, pid: int) -> bool:
        """
        Check if the running instance is healthy.
        
        Args:
            pid: Process ID to check
            
        Returns:
            True if instance is healthy, False otherwise
        """
        try:
            if not self.health_file.exists():
                return True  # Assume healthy if no health file yet
            
            # Check when health file was last updated
            health_mtime = datetime.fromtimestamp(self.health_file.stat().st_mtime)
            
            # If health file is older than 5 minutes, consider instance unhealthy
            if datetime.now() - health_mtime > timedelta(minutes=5):
                logger.warning(f"Instance PID {pid} appears unhealthy (no health update for >5 minutes)")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking instance health: {str(e)}")
            return True  # Assume healthy on error
    
    def _show_running_instance_info(self):
        """Show information about the currently running instance."""
        try:
            if not self.lock_file.exists():
                return
            
            with open(self.lock_file, 'r') as f:
                content = f.read()
            
            logger.error("Running instance details:")
            for line in content.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    logger.error(f"  {key}: {value}")
            
            # Show running time
            if self.pid_file.exists():
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                
                if psutil.pid_exists(pid):
                    process = psutil.Process(pid)
                    create_time = datetime.fromtimestamp(process.create_time())
                    running_time = datetime.now() - create_time
                    logger.error(f"  Running for: {running_time}")
            
        except Exception as e:
            logger.warning(f"Error showing running instance info: {str(e)}")
    
    def _cleanup_stale_lock(self):
        """Clean up stale lock files."""
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
            if self.pid_file.exists():
                self.pid_file.unlink()
            if self.health_file.exists():
                self.health_file.unlink()
            
            logger.info("Cleaned up stale lock files")
            
        except Exception as e:
            logger.warning(f"Error cleaning up stale lock: {str(e)}")
    
    def update_health(self):
        """Update health file to indicate instance is alive."""
        try:
            self._update_health_file()
        except Exception as e:
            logger.warning(f"Error updating health file: {str(e)}")
    
    def _update_health_file(self):
        """Update the health file with current status."""
        health_data = {
            'pid': self.current_pid,
            'last_update': datetime.now().isoformat(),
            'status': 'running',
            'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
        }
        
        with open(self.health_file, 'w') as f:
            for key, value in health_data.items():
                f.write(f"{key}:{value}\n")
    
    def cleanup(self):
        """Clean up singleton lock and associated files."""
        try:
            logger.info(f"Cleaning up singleton lock for {self.app_name}")
            
            # Remove lock files
            for file_path in [self.lock_file, self.pid_file, self.health_file]:
                if file_path.exists():
                    file_path.unlink()
                    logger.debug(f"Removed {file_path}")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals gracefully."""
        logger.info(f"Received signal {signum}, cleaning up...")
        self.cleanup()
        sys.exit(0)
    
    def get_status(self) -> dict:
        """
        Get current singleton status.
        
        Returns:
            Dictionary containing status information
        """
        status = {
            'app_name': self.app_name,
            'current_pid': self.current_pid,
            'lock_acquired': self.lock_file.exists(),
            'start_time': self.current_start_time.isoformat(),
            'lock_file': str(self.lock_file),
            'pid_file': str(self.pid_file),
            'health_file': str(self.health_file)
        }
        
        # Add running instance info if exists
        if self.lock_file.exists():
            try:
                with open(self.lock_file, 'r') as f:
                    content = f.read()
                
                status['lock_content'] = content
                
                if self.pid_file.exists():
                    with open(self.pid_file, 'r') as f:
                        status['locked_pid'] = int(f.read().strip())
                
            except Exception as e:
                status['error'] = str(e)
        
        return status

def ensure_single_instance(app_name: str = "quant_ai_trader", force: bool = False) -> SingletonManager:
    """
    Convenience function to ensure single instance.
    
    Args:
        app_name: Name of the application
        force: Force acquire lock even if another instance exists
        
    Returns:
        SingletonManager instance if successful
        
    Raises:
        RuntimeError: If another instance is already running and force=False
    """
    singleton = SingletonManager(app_name)
    
    if not singleton.acquire_lock(force=force):
        raise RuntimeError(f"Another instance of {app_name} is already running. "
                         f"Use force=True to override or stop the running instance first.")
    
    return singleton

# Global singleton instance
_global_singleton: Optional[SingletonManager] = None

def get_global_singleton() -> Optional[SingletonManager]:
    """Get the global singleton manager instance."""
    return _global_singleton

def set_global_singleton(singleton: SingletonManager):
    """Set the global singleton manager instance."""
    global _global_singleton
    _global_singleton = singleton 