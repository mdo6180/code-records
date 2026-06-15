1. Install Caddy reverse proxy:
```
brew install caddy
```
2. Edit /etc/hosts file
```
127.0.0.1 anacostia.local
```
3. Verify the reverse proxy works:
```
ping anacostia.local
```
4. Create Caddyfile in local project directory:
```
anacostia.local {
    reverse_proxy localhost:8000
}
```
5. Run FastAPI server
```
uvicorn main:app --reload
```
6. In another terminal:
```
caddy run --config Caddyfile
```
7. Browse to:
```
http://anacostia.local
```