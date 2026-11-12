import socket
import sys

HOST, PORT = "127.0.0.1", 65432
PREFIX, SUFFIX = "<<", ">>"


def serve(host=HOST, port=PORT):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen()
        print(f"Echo server is listening on {host}:{port}", file=sys.stderr)
        while True:
            conn, addr = s.accept()
            with conn, conn.makefile("rw", encoding="utf-8", newline="\n") as f:
                try:
                    for line in f:
                        f.write(f"{PREFIX}{line.rstrip(chr(10))}{SUFFIX}\n")
                        f.flush()
                except ConnectionError as e:
                    print(f"Client {addr} dropped: {e}", file=sys.stderr)


if __name__ == "__main__":
    try:
        serve()
    except KeyboardInterrupt:
        print("\nShutting down.", file=sys.stderr)