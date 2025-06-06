from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles



newline = "\n"


class App1(FastAPI):
    def __init__(self):
        super().__init__()
        self.mount("/static", StaticFiles(directory="./static1"), name="static")

        @self.get("/", response_class=HTMLResponse)
        async def home():
            return f"""
            <div id="app1-content">
                <h1>App 1 Content</h1>
                <img src="/app1/static/dod_seal.jpg" alt="App 1 Image">
            </div>
            """

class App2(FastAPI):
    def __init__(self):
        super().__init__()
        self.mount("/static", StaticFiles(directory="./static2"), name="static")

        @self.get("/", response_class=HTMLResponse)
        async def home():
            return f"""
            <div id="app2-content">
                <h1>App 2 Content</h1>
                <p>This is the content for App 2.</p>
            </div>
            """
        
class MainApp(FastAPI):
    def __init__(self, app1: App1, app2: App2):
        super().__init__()
        self.mount("/app1", app1)
        self.mount("/app2", app2)
        self.mount("/static", StaticFiles(directory="../static"), name="static")
        self.mount("/main/static", StaticFiles(directory="./static_main"), name="static_main")

        @self.get("/", response_class=HTMLResponse)
        async def home():
            return f"""
            <!DOCTYPE html>
            <html>
                <head>
                    <meta charset="UTF-8">
                    <title>Multiple StaticFiles</title>

                    <!-- Static files for app1 -->
                    <link rel="stylesheet" href="/app1/static/style.css">

                    <!-- Static files for app2 -->
                    <link rel="stylesheet" href="/app2/static/style.css">

                    <!-- Main static files -->
                    <link rel="stylesheet" href="/main/static/style.css">

                    <script src="/static/js/htmx.js" type="text/javascript"></script>
                </head>
                <body>
                    <div id="app1" hx-get="/app1" hx-trigger="click" hx-target="#snippet" hx-swap="innerHTML">App 1</div>
                    <div id="app2" hx-get="/app2" hx-trigger="click" hx-target="#snippet" hx-swap="innerHTML">App 2</div>
                    <div id="main" hx-get="/main" hx-trigger="click" hx-target="#snippet" hx-swap="innerHTML">Main</div>
                    <div id="snippet"></div>
                </body>
            </html> 
            """
        
        @self.get("/main", response_class=HTMLResponse)
        async def main():
            return f"""
            <div id="main-content">
                <h1>Main Content</h1>
                <p>This is the main content area.</p>
            </div>
            """
        
app1 = App1()
app2 = App2()
app = MainApp(app1=app1, app2=app2)
