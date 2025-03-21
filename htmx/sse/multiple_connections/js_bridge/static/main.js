document.body.addEventListener('htmx:sseOpen', (event) => {
    const source = event.detail.source;     // this is the EventSource object
    console.log(source);
});


document.body.addEventListener('htmx:sseBeforeMessage', (event) => {
    const element = event.detail.elt;   // this is the element that has the sse-connect attribute
    const event_name = event.detail.type;
    
    if (event_name === "ChangeColor") {
        event.preventDefault();     // call preventDefault() on the event to prevent the sse-swap from swapping in the data

        const data = JSON.parse(event.detail.data);
        console.log(data);

        const title = document.getElementById("title");
        title.style.color = data.color;
    }
});