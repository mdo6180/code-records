var scriptTag = document.querySelector('script[src="js/settings.js"]');
const anacostia_prefix = scriptTag.getAttribute('anacostia-prefix');

document.addEventListener('htmx:configRequest', function(evt) {
    if (evt.detail.path.startsWith("/")) {
        evt.detail.path = anacostia_prefix + evt.detail.path;
    }
});

// This file is used to configure htmx to work with a root path. 
// In this case, we are using "/ged-edap-modelsec/test-container-min-5/anacostia" as the root path. 
// This means that all htmx requests will be prefixed with this path.
// Without this configuration, htmx would try to make requests to paths like "/display", 
// which would not work because our server is actually serving content at "/ged-edap-modelsec/test-container-min-5/anacostia/display".