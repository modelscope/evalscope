/* ── Tab switching ─────────────────────────────────────────────── */
document.querySelectorAll('.tab-group').forEach(function (group) {
  group.querySelectorAll('.tab-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var target = btn.dataset.tab;

      /* Update button active state */
      group.querySelectorAll('.tab-btn').forEach(function (b) {
        b.classList.toggle('active', b === btn);
      });

      /* Show target pane; trigger Plotly resize when newly visible */
      group.querySelectorAll('.tab-pane').forEach(function (pane) {
        var wasHidden = !pane.classList.contains('active');
        pane.classList.toggle('active', pane.id === target);
        if (wasHidden && pane.classList.contains('active')) {
          window.dispatchEvent(new Event('resize'));
        }
      });
    });
  });
});
