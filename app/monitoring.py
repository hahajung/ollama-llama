from prometheus_client import start_http_server
from fastapi import Request, Response
import time

class MonitoringMiddleware:
    def __init__(self, app):
        self.app = app
        start_http_server(9000)

    async def __call__(self, request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        duration = time.time() - start_time
        request.app.state.metrics['latency'].observe(duration)
        
        if response.status_code >= 500:
            request.app.state.metrics['errors'].inc()
        
        return response