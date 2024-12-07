document.body.addEventListener('htmx:sseBeforeMessage', (event) => {
    event.preventDefault();     // call preventDefault() on the event to prevent the sse-swap from swapping in the data

    const elt = event.detail.elt;   // it's not used here, but this is the element that has the sse-connect attribute
    const data = JSON.parse(event.detail.data);
    console.log(data);

    const element = document.getElementById("title");
    element.style.color = data.color;
});