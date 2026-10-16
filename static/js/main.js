/* ============================================================
   CARFORGE — Interaksi: reveal, counter, hero reel
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {

  // ---------- 1. Reveal on scroll ----------
  const io = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('in-view');
        io.unobserve(entry.target);
      }
    });
  }, { threshold: 0.15, rootMargin: '0px 0px -40px 0px' });

  document.querySelectorAll('.reveal, .reveal-stagger').forEach((el) => io.observe(el));

  // ---------- 2. Animated counters (stats section) ----------
  const nf = new Intl.NumberFormat('en-US');

  function animateCounter(el) {
    const target = parseFloat(el.dataset.count);
    const decimals = parseInt(el.dataset.decimals || '0', 10);
    const duration = 1600;
    const start = performance.now();

    function tick(now) {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      const value = target * eased;
      el.textContent = decimals > 0
        ? value.toFixed(decimals)
        : nf.format(Math.round(value));
      if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  const counterIO = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        animateCounter(entry.target);
        counterIO.unobserve(entry.target);
      }
    });
  }, { threshold: 0.4 });

  document.querySelectorAll('[data-count]').forEach((el) => counterIO.observe(el));

  // ---------- 3. Hero reel (rotasi gambar otomatis) ----------
  const frames = Array.from(document.querySelectorAll('.reel-frame'));
  const chapterNum = document.getElementById('chapterNum');
  const fill = document.getElementById('reelProgress');

  if (frames.length > 1) {
    let index = 0;
    const DURATION = 7000;

    function showFrame(n) {
      frames.forEach((f, i) => f.classList.toggle('active', i === n));
      if (chapterNum) chapterNum.textContent = String(n + 1).padStart(2, '0');
      if (fill) {
        fill.style.transition = 'none';
        fill.style.width = '0%';
        void fill.offsetWidth; // reflow — reset animasi
        fill.style.transition = `width ${DURATION}ms linear`;
        fill.style.width = '100%';
      }
    }

    showFrame(0);
    setInterval(() => {
      index = (index + 1) % frames.length;
      showFrame(index);
    }, DURATION);
  }
});