/* ── Language switcher ─────────────────────────────────────────── */

/**
 * Apply a language to the page.
 *
 * Walks every element carrying a [data-i18n] attribute and replaces
 * its textContent with the matching translation. Falls back silently
 * when a key is absent (element keeps its original text).
 *
 * The `.desc-lang-content` handling is a safe no-op on pages that do
 * not contain description blocks (e.g. the perf report).
 *
 * @param {string} lang - Language code: 'en' | 'zh'
 */
function setLang(lang) {
  if (!I18N[lang]) return;
  var dict = I18N[lang];

  /* Update all translated elements */
  document.querySelectorAll('[data-i18n]').forEach(function (el) {
    var key = el.dataset.i18n;
    if (Object.prototype.hasOwnProperty.call(dict, key)) {
      el.textContent = dict[key];
    }
  });

  /* Show the matching description language, hide the other (eval only; no-op on perf) */
  document.querySelectorAll('.desc-lang-content').forEach(function (el) {
    el.style.display = (el.dataset.descLang === lang) ? '' : 'none';
  });

  /* Highlight the active lang button */
  document.querySelectorAll('.lang-btn').forEach(function (btn) {
    btn.classList.toggle('active', btn.dataset.lang === lang);
  });

  /* Update <html lang> for accessibility / SEO */
  document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';

  /* Persist selection across page reloads (shared key across all reports) */
  try { localStorage.setItem('evalscope-lang', lang); } catch (e) { /* ignore */ }
}

/* Bind click handlers to the EN / 中文 toggle buttons */
document.querySelectorAll('.lang-btn').forEach(function (btn) {
  btn.addEventListener('click', function () { setLang(btn.dataset.lang); });
});

/* Restore the last-selected language on page load */
(function initLang() {
  try {
    var saved = localStorage.getItem('evalscope-lang');
    if (saved && I18N[saved]) { setLang(saved); }
  } catch (e) { /* ignore */ }
}());

/* ── Dataset / run accordion arrow ───────────────────────────── */
document.querySelectorAll('details.ds-card').forEach(function (el) {
  var arrow = el.querySelector('.ds-arrow');
  function sync() {
    if (arrow) arrow.style.transform = el.open ? 'rotate(90deg)' : 'rotate(0deg)';
  }
  sync();
  el.addEventListener('toggle', function () {
    sync();
    /* Plotly responsive resize */
    if (el.open) window.dispatchEvent(new Event('resize'));
  });
});

/* ── Floating TOC toggle ─────────────────────────────────────── */
var toc       = document.getElementById('toc');
var tocToggle = document.getElementById('toc-toggle');

/* Auto-expand on wide screens */
if (window.innerWidth >= 1320) toc.classList.add('expanded');

tocToggle.addEventListener('click', function () {
  toc.classList.toggle('expanded');
});

/* Close TOC when clicking outside */
document.addEventListener('click', function (e) {
  if (!toc.contains(e.target)) toc.classList.remove('expanded');
});

/* ── TOC smooth scroll + mobile close ───────────────────────── */
document.querySelectorAll('.toc-link').forEach(function (link) {
  link.addEventListener('click', function (e) {
    var href = link.getAttribute('href');
    if (!href || !href.startsWith('#')) return;
    e.preventDefault();
    var target = document.getElementById(href.slice(1));
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      if (window.innerWidth < 1320) toc.classList.remove('expanded');
    }
  });
});

/* ── TOC active section via IntersectionObserver ─────────────── */
var tocSections = [];
document.querySelectorAll('.toc-link[data-section]').forEach(function (link) {
  var id = link.dataset.section;
  var el = document.getElementById(id);
  if (el) tocSections.push({ id: id, el: el, link: link });
});

var currentActive = null;
function setActive(id) {
  if (currentActive === id) return;
  currentActive = id;
  tocSections.forEach(function (s) {
    s.link.classList.toggle('active', s.id === id);
  });
}

if ('IntersectionObserver' in window && tocSections.length) {
  var obs = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (entry.isIntersecting) setActive(entry.target.id);
    });
  }, { rootMargin: '-15% 0px -70% 0px', threshold: 0 });

  tocSections.forEach(function (s) { obs.observe(s.el); });
}

/* ── Back-to-top visibility ──────────────────────────────────── */
var btt = document.getElementById('back-to-top');
if (btt) {
  window.addEventListener('scroll', function () {
    btt.classList.toggle('visible', window.scrollY > 300);
  }, { passive: true });

  btt.addEventListener('click', function (e) {
    e.preventDefault();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
}
