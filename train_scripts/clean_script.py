# clean_script.py
import pty
import os
import sys
import select

def main():
    argv = sys.argv[1:]
    if not argv:
        print("Usage: python clean_script.py your_script.py [args...]", file=sys.stderr)
        sys.exit(1)

    # 缓冲区：保持为 bytes
    def read(fd):
        buf = b""
        while True:
            ready, _, _ = select.select([fd], [], [], 0.01)
            if not ready:
                break
            try:
                chunk = os.read(fd, 1024)
                if not chunk:
                    break
                buf += chunk
            except OSError:
                break

        # 按行分割（保留换行符），并尝试解码
        try:
            text = buf.decode('utf-8', errors='replace')
        except Exception:
            # fallback: treat as latin1 or just pass raw
            text = buf.decode('latin1', errors='replace')

        lines = text.splitlines(keepends=True)
        for line in lines:
            if "Pre-condition Violation" not in line and "RingInfo not initialized" not in line:
                # 直接写入 stdout，不加额外换行
                sys.stdout.write(line)
                sys.stdout.flush()
        return buf  # 必须返回 bytes！

    pty.spawn(argv, read)

if __name__ == "__main__":
    main()