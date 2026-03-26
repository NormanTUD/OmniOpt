window.activeButton = null;

function startLoadingAnimation(button) {
  if (!button || !button.classList) {
    return;
  }

  window.activeButton = button;
  button.classList.add('loading');

  const textSpan = button.querySelector('.text');
  if (textSpan) {
    textSpan.textContent = 'Loading...';
  }

  if (button._loadingTimeout) {
    clearTimeout(button._loadingTimeout);
  }

  button._loadingTimeout = setTimeout(() => {
    if (window.activeButton === button) {
      stopLoadingAnimation(button);
    }
  }, 10000);
}

function stopLoadingAnimation(button) {
  if (!button || !button.classList) {
    return;
  }

  button.classList.remove('loading');

  const textSpan = button.querySelector('.text');
  if (textSpan) {
    textSpan.textContent = 'Show';
  }

  if (button._loadingTimeout) {
    clearTimeout(button._loadingTimeout);
    button._loadingTimeout = null;
  }

  if (window.activeButton === button) {
    window.activeButton = null;
  }
}
