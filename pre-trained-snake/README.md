# Build the docker image
    
`docker build battlesnake_api/ -t battlesnake:v0.2 -f Dockerfile`

# Run the docker image

`curl -v -d "" http://localhost:8080/ping`

```text
*   Trying 127.0.0.1...
* TCP_NODELAY set
* Connected to localhost (127.0.0.1) port 8080 (#0)
> POST /ping HTTP/1.1
> Host: localhost:8080
> User-Agent: curl/7.60.0
> Accept: */*
> Content-Length: 0
> Content-Type: application/x-www-form-urlencoded
>
* HTTP 1.0, assume close after body
< HTTP/1.0 200 OK
< Date: Wed, 20 Nov 2019 01:26:54 GMT
< Server: WSGIServer/0.2 CPython/3.5.2
< Content-Length: 0
< Content-Type: text/html; charset=UTF-8
<
* Closing connection 0
```