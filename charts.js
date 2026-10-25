/**
 * charts.js – Chart.js helper functions for DefendAI Dashboard
 */

const CHART_DEFAULTS = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: { legend: { labels: { color: '#8b949e', font: { size: 11 } } } },
};

const GRID_STYLE = {
  color: 'rgba(255,255,255,0.05)',
};

/** Fetch reward history and render line chart */
function loadRewardChart(canvasId) {
  fetch('/api/rewards')
    .then(r => r.json())
    .then(data => {
      if (!data.length) return;
      const labels  = data.map(d => `Ep ${d.episode}`);
      const rewards = data.map(d => d.total_reward);
      const ctx = document.getElementById(canvasId).getContext('2d');
      new Chart(ctx, {
        type: 'line',
        data: {
          labels,
          datasets: [{
            label: 'Episode Reward',
            data: rewards,
            borderColor: '#58a6ff',
            backgroundColor: 'rgba(88,166,255,0.08)',
            borderWidth: 2,
            pointRadius: 2,
            fill: true,
            tension: 0.3,
          }],
        },
        options: {
          ...CHART_DEFAULTS,
          scales: {
            x: { ticks: { color: '#8b949e', maxTicksLimit: 10 }, grid: GRID_STYLE },
            y: { ticks: { color: '#8b949e' }, grid: GRID_STYLE },
          },
        },
      });
    })
    .catch(console.error);
}

/** Fetch action distribution and render doughnut chart */
function loadActionChart(canvasId) {
  fetch('/api/actions')
    .then(r => r.json())
    .then(data => {
      const labels = Object.keys(data);
      const values = Object.values(data);
      if (!labels.length) return;
      const ctx = document.getElementById(canvasId).getContext('2d');
      new Chart(ctx, {
        type: 'doughnut',
        data: {
          labels,
          datasets: [{
            data: values,
            backgroundColor: [
              '#8b949e', '#58a6ff', '#3fb950', '#f85149', '#d29922'
            ],
            borderWidth: 1,
            borderColor: '#161b22',
          }],
        },
        options: {
          ...CHART_DEFAULTS,
          cutout: '65%',
        },
      });
    })
    .catch(console.error);
}

/** Initialise live reward sparkline (returns chart instance) */
function initLiveChart(canvasId) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Live Reward',
        data: [],
        borderColor: '#3fb950',
        backgroundColor: 'rgba(63,185,80,0.08)',
        borderWidth: 2,
        pointRadius: 1,
        fill: true,
        tension: 0.3,
      }],
    },
    options: {
      ...CHART_DEFAULTS,
      animation: { duration: 0 },
      scales: {
        x: { ticks: { color: '#8b949e', maxTicksLimit: 10 }, grid: GRID_STYLE },
        y: { ticks: { color: '#8b949e' }, grid: GRID_STYLE },
      },
    },
  });
}

/** Push new data point to live chart */
function updateLiveChart(chart, rewards) {
  chart.data.labels = rewards.map((_, i) => i + 1);
  chart.data.datasets[0].data = rewards;
  chart.update('none');
}
