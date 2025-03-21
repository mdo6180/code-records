document.body.addEventListener('htmx:sseOpen', (event) => {
    const source = event.detail.source;
    console.log(source);
});


document.body.addEventListener('htmx:sseBeforeMessage', (event) => {
    const elt = event.detail.elt;   // this is the element that has the sse-connect attribute
    const event_name = event.detail.type;
    
    if (event_name === "ChangeColor") {
        event.preventDefault();     // call preventDefault() on the event to prevent the sse-swap from swapping in the data

        const data = JSON.parse(event.detail.data);
        console.log(data);

        const element = document.getElementById("title");
        element.style.color = data.color;
    }
});