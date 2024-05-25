def homepage():
    return """<html>
        <head>
            <meta charset="UTF-8">
            <title>Anacostia Console</title>
            
            <!-- non-minified Htmx -->
            <script src="/static/js/htmx.js" type="text/javascript"></script>

            <!-- htmx SSE extension -->
            <script src="/static/js/sse.js"></script>
        </head>
        <body hx-ext="sse" sse-connect="/event-source">
            <div hx-get="/chat-row" hx-trigger="load, sse:EventName" hx-swap="beforeend"></div>
        </body>
    </html>
    """

def chat_data(i: int):
    return f"""
    <div>Content to swap into your HTML page ({i}).</div>
    <div>More content to swap into your HTML page ({i}).</div>
    """