import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Tuple
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("email_service")


class EmailService:
    """
    Handles sending transactional emails (email verification and password reset).
    Supports SMTP over TLS (port 587), SSL (port 465), or plaintext.
    """

    @classmethod
    def get_smtp_config(cls, db=None) -> dict:
        """
        Retrieves SMTP configuration from environment or database system settings.
        """
        host = settings.SMTP_HOST
        port = settings.SMTP_PORT
        user = settings.SMTP_USER
        password = settings.SMTP_PASSWORD
        from_email = settings.SMTP_FROM_EMAIL
        use_tls = settings.SMTP_USE_TLS

        if db:
            try:
                from app.db import crud
                db_host = crud.get_system_setting(db, "SMTP_HOST")
                if db_host: host = db_host
                db_port = crud.get_int_setting(db, "SMTP_PORT", default=port)
                if db_port: port = db_port
                db_user = crud.get_system_setting(db, "SMTP_USER")
                if db_user: user = db_user
                db_pass = crud.get_system_setting(db, "SMTP_PASSWORD")
                if db_pass: password = db_pass
                db_from = crud.get_system_setting(db, "SMTP_FROM_EMAIL")
                if db_from: from_email = db_from
                db_tls = crud.get_bool_setting(db, "SMTP_USE_TLS", default=use_tls)
                use_tls = db_tls
            except Exception as e:
                logger.warning(f"Error loading DB SMTP config: {e}")

        is_configured = bool(host and user and password)
        return {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "from_email": from_email or user,
            "use_tls": use_tls,
            "is_configured": is_configured
        }

    @classmethod
    def is_smtp_configured(cls, db=None) -> bool:
        cfg = cls.get_smtp_config(db)
        return cfg["is_configured"]

    @classmethod
    def send_email(cls, to_email: str, subject: str, html_body: str, text_body: str, db=None) -> Tuple[bool, str]:
        """
        Sends an email using configured SMTP settings.
        Returns: (success: bool, message: str)
        """
        cfg = cls.get_smtp_config(db)

        if not cfg["is_configured"]:
            logger.info(f"[DEV MODE] Email to {to_email} with subject '{subject}':\n{text_body}")
            return False, "SMTP is not configured. (To receive real emails, configure SMTP_USER and SMTP_PASSWORD in .env or Admin Dashboard)."

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            # Format with branded display name (e.g. "DocuMind AI <no-reply@...>")
            from_addr = cfg["from_email"]
            if "<" in from_addr:
                msg["From"] = from_addr
            else:
                msg["From"] = f"DocuMind AI <{from_addr}>"
            msg["To"] = to_email

            part1 = MIMEText(text_body, "plain")
            part2 = MIMEText(html_body, "html")
            msg.attach(part1)
            msg.attach(part2)

            port = int(cfg["port"])
            if port == 465:
                # SSL Connection
                server = smtplib.SMTP_SSL(cfg["host"], port, timeout=15)
            else:
                # Standard / STARTTLS Connection
                server = smtplib.SMTP(cfg["host"], port, timeout=15)
                if cfg["use_tls"]:
                    server.starttls()

            if cfg["user"] and cfg["password"]:
                server.login(cfg["user"], cfg["password"])

            server.sendmail(cfg["from_email"], to_email, msg.as_string())
            server.quit()
            logger.info(f"Successfully sent email to {to_email}")
            return True, "Email sent successfully."
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False, f"Email delivery failed: {str(e)}"

    @classmethod
    def send_verification_email(cls, to_email: str, name: str, token: str, db=None) -> Tuple[bool, str]:
        subject = "Verify Your DocuMind RAG Account"
        text_body = f"""Hello {name},

Thank you for registering with DocuMind Multi-User RAG.

Your 6-digit verification code is: {token}

This code will expire in 10 minutes. Please enter this code in the application to complete your registration.

Best regards,
DocuMind Team
"""
        html_body = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 24px; border: 1px solid #e2e8f0; border-radius: 10px; background-color: #ffffff;">
            <div style="text-align: center; margin-bottom: 20px;">
                <h2 style="color: #2563eb; margin: 0;">🧠 DocuMind AI</h2>
                <p style="color: #64748b; font-size: 14px; margin: 4px 0 0 0;">Multi-User Document Intelligence Platform</p>
            </div>
            <p style="color: #334155; font-size: 15px;">Hello <b>{name}</b>,</p>
            <p style="color: #334155; font-size: 15px;">Thank you for registering with DocuMind. Please use the 6-digit verification code below to verify your email and activate your account:</p>
            <div style="background-color: #f8fafc; border: 2px dashed #cbd5e1; padding: 18px; text-align: center; border-radius: 8px; font-size: 32px; font-weight: bold; letter-spacing: 8px; color: #1e293b; margin: 24px 0;">
                {token}
            </div>
            <p style="color: #ef4444; font-size: 13px; font-weight: 500;">⏱️ This verification code will expire in 10 minutes.</p>
            <p style="color: #94a3b8; font-size: 12px; margin-top: 24px; border-top: 1px solid #f1f5f9; padding-top: 12px;">If you did not create an account with DocuMind, you can safely ignore this email.</p>
        </div>
        """
        return cls.send_email(to_email, subject, html_body, text_body, db=db)

    @classmethod
    def send_password_reset_email(cls, to_email: str, name: str, token: str, db=None) -> Tuple[bool, str]:
        subject = "Reset Your DocuMind RAG Password"
        text_body = f"""Hello {name},

We received a request to reset your password.

Your password reset code is: {token}

This code will expire in 10 minutes.

Best regards,
DocuMind Team
"""
        html_body = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 24px; border: 1px solid #e2e8f0; border-radius: 10px; background-color: #ffffff;">
            <div style="text-align: center; margin-bottom: 20px;">
                <h2 style="color: #2563eb; margin: 0;">🧠 DocuMind AI</h2>
                <p style="color: #64748b; font-size: 14px; margin: 4px 0 0 0;">Password Reset Request</p>
            </div>
            <p style="color: #334155; font-size: 15px;">Hello <b>{name}</b>,</p>
            <p style="color: #334155; font-size: 15px;">We received a request to reset your account password. Use the code below to set a new password:</p>
            <div style="background-color: #f8fafc; border: 2px dashed #cbd5e1; padding: 18px; text-align: center; border-radius: 8px; font-size: 32px; font-weight: bold; letter-spacing: 8px; color: #1e293b; margin: 24px 0;">
                {token}
            </div>
            <p style="color: #ef4444; font-size: 13px; font-weight: 500;">⏱️ This reset code will expire in 10 minutes.</p>
            <p style="color: #94a3b8; font-size: 12px; margin-top: 24px; border-top: 1px solid #f1f5f9; padding-top: 12px;">If you did not request this, please disregard this email.</p>
        </div>
        """
        return cls.send_email(to_email, subject, html_body, text_body, db=db)
