from typing import Optional
from sqlalchemy.orm import Session
from app.db import crud
from app.utils.logger import get_logger

logger = get_logger("audit_service")


class AuditService:
    """
    Helper for recording security, administrative, and user events into audit_logs table.
    """

    @staticmethod
    def log_event(
        db: Session,
        action: str,
        user_id: Optional[int] = None,
        details: Optional[str] = None,
        ip_address: Optional[str] = None
    ):
        try:
            crud.create_audit_log(
                db=db,
                action=action,
                user_id=user_id,
                details=details,
                ip_address=ip_address
            )
            logger.info(f"AUDIT [{action}] (User: {user_id}): {details}")
        except Exception as e:
            logger.error(f"Failed to record audit log: {e}")
