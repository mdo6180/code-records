import { greet } from '/static/utils.js';


document.body.addEventListener("click", (event) => {
    const message = greet('FastAPI');
    alert(message);
});