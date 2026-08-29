import uvicorn
import argparse
from app.config import settings


def main():
    parser = argparse.ArgumentParser(description="Start the RAG Chatbot FastAPI Server")
    parser.add_argument("--host", type=str, default=settings.HOST, help="Host interface to bind")
    parser.add_argument("--port", type=int, default=settings.PORT, help="Port to bind")
    parser.add_argument("--reload", action="store_true", default=settings.DEBUG, help="Enable auto-reload on code change")

    args = parser.parse_args()

    print(f"🚀 Starting server on http://{args.host}:{args.port} (reload={args.reload})")
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
