/* Apply the shared console theme before first paint. */
(function () {
  'use strict';

  var theme = 'dark';
  try {
    var saved = localStorage.getItem('evalscope-theme');
    if (saved === 'light' || saved === 'dark') {
      theme = saved;
    } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
      theme = 'light';
    }
  } catch (e) {
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
      theme = 'light';
    }
  }
  document.documentElement.setAttribute('data-theme', theme);
}());
