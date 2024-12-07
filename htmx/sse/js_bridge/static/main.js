document.body.addEventListener('htmx:sseBeforeMessage', (event) => {
    event.preventDefault();     // i call preventDefault() on the event to prevent the sse-swap from swapping in the data

    const elt = event.detail.elt;   // i don't use it here, but this is the element that has the sse-connect attribute
    const data = event.detail.data;
    console.log(data);

    const element = document.getElementById("title");
    element.style.color = data;
});