def homepage():
    return """
    <!DOCTYPE>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>Anacostia Console</title>

            <!-- minified _Hyperscript (Whole 9 Yards version) -->
            <script src="/static/js/_hyperscript_w9y.min.js"></script>

            <style>
                .red {
                    color: red;
                }
                .green {
                    color: green;
                }
            </style>
        </head>
        <body>
            <div _="eventsource EventStream from /event-source
                        on open
                            log 'event source connected'
                        end
                    end">
                <div _="on HelloEvent from EventStream 
                            toggle .red on me
                            log event.data
                        end">event1</div>
                <div _="on GoodbyeEvent from EventStream 
                            toggle .green on me
                            log event.data
                        end">event2</div>
            </div>
        </body>
    </html>
    """
