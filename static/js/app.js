/* global Chart */

console.log("TNEB Smart UI loaded - app.js v1001");

(() => {
  const state = {
    meta: null,
    charts: {},
    activePage: "home",
  };

  const $ = (id) => document.getElementById(id);

  const palette = [
    "#FFB300", // Solar Gold
    "#00C853", // Emerald
    "#FF3D00", // Alert Orange
    "#8D6E63", // Warm Brown
    "#FFC107", // Amber
    "#6D4C41", // Espresso
    "#FFD54F", // Soft Gold
    "#43A047", // Green
  ];

  // =====================================================
  // Formatting helpers
  // =====================================================

  function formatKwh(value, digits = 1) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "--";
    }

    return `${Number(value).toLocaleString("en-IN", {
      maximumFractionDigits: digits,
      minimumFractionDigits: digits,
    })} kWh`;
  }

  function formatUnits(value, digits = 1) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "--";
    }

    return `${Number(value).toLocaleString("en-IN", {
      maximumFractionDigits: digits,
      minimumFractionDigits: digits,
    })} units`;
  }

  function formatEnergy(value, digits = 1) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "--";
    }

    const formatted = Number(value).toLocaleString("en-IN", {
      maximumFractionDigits: digits,
      minimumFractionDigits: digits,
    });

    return `${formatted} units / ${formatted} kWh`;
  }

  function formatRs(value, digits = 0) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "₹--";
    }

    return `₹${Number(value).toLocaleString("en-IN", {
      maximumFractionDigits: digits,
      minimumFractionDigits: digits,
    })}`;
  }

  // =====================================================
  // API helper
  // =====================================================

  async function fetchJSON(url, options = {}) {
    const response = await fetch(url, options);

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `Request failed: ${response.status}`);
    }

    return response.json();
  }

  // =====================================================
  // Chart helpers
  // =====================================================

  function destroyChart(id) {
    if (state.charts[id]) {
      state.charts[id].destroy();
      delete state.charts[id];
    }
  }

  function lineChart(id, labels, datasets) {
    const canvas = $(id);

    if (!canvas || typeof Chart === "undefined") {
      return;
    }

    destroyChart(id);

    state.charts[id] = new Chart(canvas.getContext("2d"), {
      type: "line",
      data: {
        labels,
        datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: "index",
          intersect: false,
        },
        plugins: {
          legend: {
            position: "bottom",
            labels: {
              usePointStyle: true,
              boxWidth: 10,
            },
          },
        },
        scales: {
          x: {
            grid: {
              display: false,
            },
            ticks: {
              maxTicksLimit: 8,
            },
          },
          y: {
            beginAtZero: true,
            grid: {
              color: "rgba(148,163,184,0.18)",
            },
          },
        },
      },
    });
  }

  function barChart(id, labels, data, label, color = "#FFB300") {
    const canvas = $(id);

    if (!canvas || typeof Chart === "undefined") {
      return;
    }

    destroyChart(id);

    state.charts[id] = new Chart(canvas.getContext("2d"), {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label,
            data,
            backgroundColor: color,
            borderRadius: 10,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false,
          },
        },
        scales: {
          x: {
            grid: {
              display: false,
            },
            ticks: {
              maxTicksLimit: 8,
            },
          },
          y: {
            beginAtZero: true,
            grid: {
              color: "rgba(148,163,184,0.18)",
            },
          },
        },
      },
    });
  }

  function pieChart(id, labels, data) {
    const canvas = $(id);

    if (!canvas || typeof Chart === "undefined") {
      return;
    }

    destroyChart(id);

    state.charts[id] = new Chart(canvas.getContext("2d"), {
      type: "doughnut",
      data: {
        labels,
        datasets: [
          {
            data,
            backgroundColor: palette,
            borderWidth: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "62%",
        plugins: {
          legend: {
            position: "bottom",
            labels: {
              usePointStyle: true,
              boxWidth: 10,
            },
          },
        },
      },
    });
  }

  function stackedBarChart(id, labels, datasets) {
    const canvas = $(id);

    if (!canvas || typeof Chart === "undefined") {
      return;
    }

    destroyChart(id);

    state.charts[id] = new Chart(canvas.getContext("2d"), {
      type: "bar",
      data: {
        labels,
        datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: {
              usePointStyle: true,
              boxWidth: 10,
            },
          },
        },
        scales: {
          x: {
            stacked: true,
            grid: {
              display: false,
            },
            ticks: {
              maxTicksLimit: 10,
            },
          },
          y: {
            stacked: true,
            beginAtZero: true,
            grid: {
              color: "rgba(148,163,184,0.18)",
            },
          },
        },
      },
    });
  }

  // =====================================================
  // Date helpers
  // =====================================================

  function dateMinus(dateString, days) {
    const d = new Date(`${dateString}T00:00:00`);
    d.setDate(d.getDate() - days);
    return d.toISOString().slice(0, 10);
  }

  function nextDate(dateString) {
    const d = new Date(`${dateString}T00:00:00`);
    d.setDate(d.getDate() + 1);
    return d.toISOString().slice(0, 10);
  }

  // =====================================================
  // Navigation
  // =====================================================

  function setPage(page) {
    state.activePage = page;

    document.querySelectorAll(".page").forEach((section) => {
      section.classList.remove("active");
    });

    const target = $(`page-${page}`);

    if (target) {
      target.classList.add("active");
    }

    document.querySelectorAll(".nav-item").forEach((button) => {
      button.classList.toggle("active", button.dataset.page === page);
    });

    const titles = {
      home: [
        "Electricity Usage & Bill Forecast",
        "Simple household view for current units, month-end prediction, and bill estimate.",
      ],
      analytics: [
        "Analytical Dashboard",
        "Complete day-wise, week-wise, year-wise, and appliance-wise electricity story.",
      ],
      features: [
        "Smart Features",
        "Slab tracking, forecast range, anomaly detection, saving tips and downloadable reports.",
      ],
      realtime: [
        "Live Daily Check",
        "Add manual appliance readings and instantly project usage, bill, and alerts.",
      ],
    };

   if ($("pageTitle")) {
  $("pageTitle").textContent = titles[page][0];
}

if ($("pageSubtitle")) {
  $("pageSubtitle").textContent = titles[page][1];
}

/* Hide only the title text on Live Check page, but keep its space
   so the right-side icons/buttons stay on the right */
if ($("pageTitle") && $("pageSubtitle")) {
  if (page === "realtime") {
    $("pageTitle").style.visibility = "hidden";
    $("pageSubtitle").style.visibility = "hidden";
  } else {
    $("pageTitle").style.visibility = "visible";
    $("pageSubtitle").style.visibility = "visible";
  }
}

    if (window.innerWidth <= 1100 && $("sidebar")) {
      $("sidebar").classList.remove("open");
    }

    if (page === "analytics") {
      loadAnalytics().catch(console.error);
    }

    if (page === "features") {
      loadFeatures().catch(console.error);
    }
  }

  // =====================================================
  // Bill table
  // =====================================================

  function renderBillBreakdown(rows, tableId = "billBreakdownTable") {
    const table = $(tableId);

    if (!table) {
      return;
    }

    table.innerHTML = `
      <thead>
        <tr>
          <th>Slab</th>
          <th>Units</th>
          <th>Rate</th>
          <th>Amount</th>
        </tr>
      </thead>
    `;

    const body = document.createElement("tbody");

    rows.forEach((row) => {
      const tr = document.createElement("tr");

      tr.innerHTML = `
        <td>${row.slab}</td>
        <td>${row.units}</td>
        <td>₹${row.rate}</td>
        <td>${formatRs(row.amount, 2)}</td>
      `;

      body.appendChild(tr);
    });

    table.appendChild(body);
  }

  // =====================================================
  // Load metadata
  // =====================================================

  async function loadMeta() {
    const meta = await fetchJSON("/api/meta");

    state.meta = meta;

    if ($("metaRange")) {
      $("metaRange").innerHTML = `${meta.dataStart} → ${meta.dataEnd}<br>${meta.rows} daily readings`;
    }

    if ($("modelStatus")) {
      $("modelStatus").textContent =
        meta.modelStatus === "loaded" ? "Model ready" : "Model trained";
    }

    if ($("cycleStart")) {
      $("cycleStart").value = dateMinus(meta.dataEnd, 59);
    }

    if ($("analyticsStart")) {
      $("analyticsStart").value = meta.dataStart;
    }

    if ($("analyticsEnd")) {
      $("analyticsEnd").value = meta.dataEnd;
    }

    if ($("rtDate")) {
      $("rtDate").value = nextDate(meta.dataEnd);
    }

    if ($("rtCycleStart")) {
      $("rtCycleStart").value = dateMinus(meta.dataEnd, 29);
    }
  }

  // =====================================================
  // Home / overview
  // =====================================================

  async function loadOverview() {
    const cycleStart = $("cycleStart") ? $("cycleStart").value : "";

    const data = await fetchJSON(`/api/overview?cycle_start=${cycleStart}`);

    $("latestUnits").textContent = formatEnergy(data.latestUnits);
    $("latestDate").textContent = `As of ${data.latestDate}`;

    $("usedMonth").textContent = formatEnergy(data.usedThisMonth);

    $("monthEndUnits").textContent = formatEnergy(data.projectedMonthUnits);
    $("monthEndDate").textContent = `Forecast to ${data.monthEnd}`;

    $("cycleUnits").textContent = formatEnergy(data.projectedCycleUnits);
    $("cycleDates").textContent = `${data.cycleStart} → ${data.cycleEnd}`;

    $("projectedBillHero").textContent = formatRs(data.projectedBill);
    $("projectedBillHelp").textContent = `${formatEnergy(data.projectedCycleUnits)} projected for current 2-month cycle`;

    const alert = data.alert;
    const alertBox = $("mainAlert");

    if (alertBox) {
      alertBox.className = `alert-card ${alert.level}`;
      alertBox.innerHTML = `<b>${alert.title}:</b> ${alert.message}`;
    }

    $("cycleDay").textContent = `${data.cycleDay}/60`;

    $("progressRing").style.setProperty(
      "--pct",
      Math.round((data.cycleDay / 60) * 100)
    );

    $("cycleUsed").textContent = formatEnergy(data.cycleUsedSoFar);
    $("cycleRemaining").textContent = formatEnergy(data.cycleForecastRemaining);
    $("historicalCompare").textContent = `${data.differencePct > 0 ? "+" : ""}${data.differencePct}%`;

    renderBillBreakdown(data.billBreakdown);

    const labels = data.monthChart.map((d) => d.date);

    const actual = data.monthChart.map((d) =>
      d.type === "Actual" ? d.units : null
    );

    const forecast = data.monthChart.map((d) =>
      d.type === "Forecast" ? d.units : null
    );

    lineChart("monthForecastChart", labels, [
      {
        label: "Actual",
        data: actual,
        borderColor: "#FFB300",
        backgroundColor: "rgba(255,179,0,0.14)",
        tension: 0.35,
        pointRadius: 2,
        spanGaps: false,
      },
      {
        label: "Forecast",
        data: forecast,
        borderColor: "#00C853",
        backgroundColor: "rgba(0,200,83,0.12)",
        tension: 0.35,
        pointRadius: 2,
        borderDash: [6, 6],
        spanGaps: false,
      },
    ]);
  }

  // =====================================================
  // Analytics
  // =====================================================

  async function loadAnalytics() {
    const start = $("analyticsStart") ? $("analyticsStart").value : "";
    const end = $("analyticsEnd") ? $("analyticsEnd").value : "";

    const data = await fetchJSON(`/api/analytics?start=${start}&end=${end}`);

    const s = data.summary;

    $("anaTotal").textContent = formatEnergy(s.totalUnits, 0);
    $("anaAvg").textContent = formatEnergy(s.avgDaily, 1);
    $("anaPeak").textContent = formatEnergy(s.peakUnits, 1);
    $("anaPeakDate").textContent = s.peakDate;
    $("anaTopApp").textContent = s.topAppliance;

    if (s.summerAvg !== null && s.nonSummerAvg !== null) {
      const diff = (s.summerAvg - s.nonSummerAvg).toFixed(1);

      $("storyInsight").innerHTML = `
        📌 Summer days average <b>${formatEnergy(s.summerAvg)}</b>,
        while non-summer days average <b>${formatEnergy(s.nonSummerAvg)}</b>.
        Difference: <b>${diff > 0 ? "+" : ""}${diff} units/day</b>.
      `;
    } else {
      $("storyInsight").textContent =
        "Not enough summer and non-summer data in this range for comparison.";
    }

    lineChart("dailyTrendChart", data.daily.map((d) => d.date), [
      {
        label: "Daily units",
        data: data.daily.map((d) => d.units),
        borderColor: "#FFB300",
        backgroundColor: "rgba(255,179,0,0.14)",
        tension: 0.3,
        pointRadius: 0,
      },
      {
        label: "7-day average",
        data: data.daily.map((d) => d.rolling7),
        borderColor: "#00C853",
        backgroundColor: "rgba(0,200,83,0.12)",
        tension: 0.3,
        pointRadius: 0,
        borderWidth: 3,
      },
    ]);

    barChart(
      "dayOfWeekChart",
      data.dayOfWeek.map((d) => d.day),
      data.dayOfWeek.map((d) => d.units),
      "Avg units",
      "#00C853"
    );

    barChart(
      "weeklyChart",
      data.weekly.map((d) => d.date),
      data.weekly.map((d) => d.units),
      "Weekly units",
      "#FFB300"
    );

    barChart(
      "monthlyChart",
      data.monthly.map((d) => d.month),
      data.monthly.map((d) => d.units),
      "Monthly units",
      "#8D6E63"
    );

    pieChart(
      "appliancePieChart",
      data.applianceTotals.map((d) => d.appliance),
      data.applianceTotals.map((d) => d.units)
    );

    if (data.applianceMonthly.length) {
      const labels = data.applianceMonthly.map((d) => d.month);

      const applianceNames = Object.keys(data.applianceMonthly[0]).filter(
        (k) => k !== "month"
      );

      const datasets = applianceNames.map((name, index) => ({
        label: name,
        data: data.applianceMonthly.map((d) => d[name]),
        backgroundColor: palette[index % palette.length],
        borderRadius: 6,
      }));

      stackedBarChart("applianceStackChart", labels, datasets);
    }

    renderHeatmap(data.heatmap);

    destroyChart("billingChart");

    const billCanvas = $("billingChart");

    if (billCanvas && typeof Chart !== "undefined") {
      state.charts.billingChart = new Chart(billCanvas.getContext("2d"), {
        type: "bar",
        data: {
          labels: data.billCycles.map((d) => `C${d.cycle}`),
          datasets: [
            {
              label: "Units",
              data: data.billCycles.map((d) => d.units),
              backgroundColor: "#FFB300",
              borderRadius: 10,
              yAxisID: "y",
            },
            {
              label: "Bill ₹",
              data: data.billCycles.map((d) => d.bill),
              backgroundColor: "#FF3D00",
              borderRadius: 10,
              yAxisID: "y1",
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: "bottom",
            },
          },
          scales: {
            x: {
              grid: {
                display: false,
              },
            },
            y: {
              beginAtZero: true,
              position: "left",
              title: {
                display: true,
                text: "Units",
              },
            },
            y1: {
              beginAtZero: true,
              position: "right",
              grid: {
                drawOnChartArea: false,
              },
              title: {
                display: true,
                text: "₹",
              },
            },
          },
        },
      });
    }
  }

  // =====================================================
  // Heatmap
  // =====================================================

  function renderHeatmap(items) {
    const container = $("heatmap");

    if (!container) {
      return;
    }

    container.innerHTML = "";

    if (!items.length) {
      return;
    }

    const max = Math.max(...items.map((d) => d.units));
    const groups = {};

    items.forEach((d) => {
      if (!groups[d.month]) {
        groups[d.month] = {};
      }

      groups[d.month][d.day] = d.units;
    });

    Object.entries(groups).forEach(([month, dayMap]) => {
      const row = document.createElement("div");
      row.className = "heat-row";

      const label = document.createElement("div");
      label.className = "heat-label";
      label.textContent = month;

      row.appendChild(label);

      for (let day = 1; day <= 31; day++) {
        const cell = document.createElement("div");
        cell.className = "heat-cell";

        const units = dayMap[day];

        if (units !== undefined) {
          const intensity = max ? units / max : 0;
          cell.style.background = `rgba(255, 179, 0, ${0.12 + intensity * 0.88})`;
          cell.dataset.tip = `${month} ${day}: ${units} units`;
        } else {
          cell.style.opacity = 0.15;
        }

        row.appendChild(cell);
      }

      container.appendChild(row);
    });
  }

  // =====================================================
  // Features page
  // =====================================================

  async function loadFeatures() {
    const cycleStart = $("cycleStart") ? $("cycleStart").value : "";

    const data = await fetchJSON(`/api/features?cycle_start=${cycleStart}`);

    $("featUnitsToSlab").textContent =
      data.slabInfo.nextThreshold ? formatUnits(data.slabInfo.unitsRemaining) : "Above 1000";

    $("featSlabText").textContent = data.slabInfo.message;

    $("featRange").textContent =
      `${formatUnits(data.forecastRange.lowerUnits)} – ${formatUnits(data.forecastRange.upperUnits)}`;

    $("featBillRange").textContent =
      `${formatRs(data.forecastRange.lowerBill)} – ${formatRs(data.forecastRange.upperBill)}`;

    $("featAnomaly").textContent = data.anomaly.status;

    $("featAnomalyText").textContent = data.anomaly.threshold
      ? `Latest ${data.anomaly.latestUnits} units, threshold ${data.anomaly.threshold} units`
      : "Not enough history";

    renderBillBreakdown(data.billBreakdown, "featureBillBreakdown");

    barChart(
      "slabTrackerChart",
      ["Projected", "500", "600", "800", "1000"],
      [data.cycle.projectedUnits, 500, 600, 800, 1000],
      "Units",
      "#FFB300"
    );

    barChart(
      "forecastRangeChart",
      ["Lower", "Expected", "Upper"],
      [
        data.forecastRange.lowerUnits,
        data.forecastRange.expectedUnits,
        data.forecastRange.upperUnits,
      ],
      "Units",
      "#00C853"
    );

    barChart(
      "applianceSavingsChart",
      data.applianceInsights.opportunities.map((d) => d.appliance),
      data.applianceInsights.opportunities.map((d) => d.estimatedUnitsSaved),
      "Potential units saved",
      "#00C853"
    );

    lineChart(
      "recentUsageChart",
      data.recentUsage.map((d) => d.date),
      [
        {
          label: "Recent units",
          data: data.recentUsage.map((d) => d.units),
          borderColor: "#FFB300",
          backgroundColor: "rgba(255,179,0,0.14)",
          tension: 0.3,
          pointRadius: 0,
        },
      ]
    );

    const list = $("recommendationsList");
    list.innerHTML = "";

    data.recommendations.forEach((r) => {
      const el = document.createElement("div");
      el.className = "recommendation-card";

      el.innerHTML = `
        <b>${r.priority}: ${r.title}</b>
        <p>${r.message}</p>
        <div class="recommendation-meta">
          <span>Potential: ${formatUnits(r.impactUnits || 0)}</span>
          <span>Approx: ${formatRs(r.impactRs || 0)}</span>
        </div>
      `;

      list.appendChild(el);
    });

    const applianceBox = $("applianceInsightTable");
    applianceBox.innerHTML = "";

    data.applianceInsights.totals.slice(0, 5).forEach((a) => {
      const el = document.createElement("div");
      el.className = "mini-item";

      el.innerHTML = `
        <b>${a.appliance}</b>: ${formatUnits(a.units)}
        (${a.share}%).
        Trend vs previous 30 days:
        ${a.trend > 0 ? "+" : ""}${a.trend} units
      `;

      applianceBox.appendChild(el);
    });

    const anomalyBox = $("anomalyBox");

    anomalyBox.innerHTML = data.anomaly.threshold
      ? `
        <b>${data.anomaly.status}</b><br>
        Latest: ${formatUnits(data.anomaly.latestUnits)}<br>
        30-day normal threshold: ${formatUnits(data.anomaly.threshold)}
      `
      : `<b>${data.anomaly.status}</b>`;

    const rep = data.monthlyReport;

    $("monthlyReportBox").innerHTML = `
      <div class="report-item">
        <span>Period</span>
        <b>${rep.start} → ${rep.end}</b>
      </div>

      <div class="report-item">
        <span>Total</span>
        <b>${formatEnergy(rep.totalUnits)}</b>
      </div>

      <div class="report-item">
        <span>Avg Daily</span>
        <b>${formatEnergy(rep.avgDaily)}</b>
      </div>

      <div class="report-item">
        <span>Peak</span>
        <b>${formatEnergy(rep.peakUnits)}</b>
      </div>
    `;
  }

  // =====================================================
  // Manual appliance input
  // =====================================================

  function calculateApplianceTotal() {
    let total = 0;
    const appliances = {};

    document.querySelectorAll(".appliance-unit").forEach((input) => {
      const key = input.dataset.key;
      const value = Number(input.value || 0);

      appliances[key] = value;
      total += value;
    });

    const rounded = Math.round(total * 10) / 10;

    if ($("applianceTotal")) {
      $("applianceTotal").textContent =
        `${rounded.toFixed(1)} units / ${rounded.toFixed(1)} kWh`;
    }

    if ($("rtUnits")) {
      $("rtUnits").value = rounded.toFixed(1);
    }

    return {
      total: rounded,
      appliances,
    };
  }

  function setupApplianceInputs() {
    document.querySelectorAll(".appliance-unit").forEach((input) => {
      input.addEventListener("input", calculateApplianceTotal);
    });

    calculateApplianceTotal();
  }

  // =====================================================
  // Realtime
  // =====================================================

  async function runRealtime(event) {
    event.preventDefault();

    const resultBox = $("liveResult");
    const msgBox = $("rtMessages");

    try {
      if (msgBox) {
        msgBox.innerHTML = "";
      }

      const date = $("rtDate").value;
      const cycleStart = $("rtCycleStart").value;

      const applianceResult = calculateApplianceTotal();
      const totalUnits = applianceResult.total;

      if (!date) {
        throw new Error("Please select today's date.");
      }

      if (totalUnits <= 0) {
        throw new Error("Please enter at least one appliance unit value.");
      }

      if (resultBox) {
        resultBox.innerHTML = "<p>Running prediction...</p>";
      }

      const data = await fetchJSON("/api/realtime", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          date,
          units: totalUnits,
          cycleStart,
          appliances: applianceResult.appliances,
        }),
      });

      if (msgBox) {
        msgBox.innerHTML = "";

        if (data.messages && data.messages.length) {
          data.messages.forEach((message) => {
            const el = document.createElement("div");
            el.className = "message";
            el.textContent = message;
            msgBox.appendChild(el);
          });
        }
      }

      const anomaly = data.anomaly || {};
      const cycle = data.cycle;

      if (resultBox) {
        resultBox.innerHTML = `
          <div class="live-grid">
            <div class="live-item">
              <span>Today</span>
              <b>${formatEnergy(data.todayUnits)}</b>
            </div>

            <div class="live-item">
              <span>Tomorrow</span>
              <b>${formatEnergy(data.tomorrowPrediction)}</b>
            </div>

            <div class="live-item">
              <span>Next 60 Days</span>
              <b>${formatEnergy(data.next60Units)}</b>
            </div>

            <div class="live-item">
              <span>Estimated Bill</span>
              <b>${formatRs(data.next60Bill)}</b>
            </div>

            <div class="live-item">
              <span>Anomaly</span>
              <b>${anomaly.status || "--"}</b>
            </div>

            <div class="live-item">
              <span>Alert</span>
              <b>${data.alert.title}</b>
            </div>

            ${
              cycle
                ? `
                  <div class="live-item">
                    <span>Cycle Day</span>
                    <b>${cycle.cycleDay}/60</b>
                  </div>

                  <div class="live-item">
                    <span>Projected Cycle Bill</span>
                    <b>${formatRs(cycle.projectedBill)}</b>
                  </div>
                `
                : ""
            }
          </div>
        `;
      }

      if (data.next60Forecast && data.next60Forecast.length) {
        lineChart(
          "realtimeForecastChart",
          data.next60Forecast.map((d) => d.date),
          [
            {
              label: "Forecast units",
              data: data.next60Forecast.map((d) => d.units),
              borderColor: "#FFB300",
              backgroundColor: "rgba(255,179,0,0.14)",
              tension: 0.35,
              pointRadius: 0,
            },
          ]
        );
      }
    } catch (error) {
      console.error(error);

      if (resultBox) {
        resultBox.innerHTML = `
          <div class="alert-card warning">
            <b>Something went wrong:</b> ${error.message}
          </div>
        `;
      }
    }
  }

  // =====================================================
  // Events
  // =====================================================

  function setupEvents() {
    document.querySelectorAll(".nav-item").forEach((button) => {
      button.addEventListener("click", () => {
        setPage(button.dataset.page);
      });
    });

    if ($("hamburger")) {
      $("hamburger").addEventListener("click", () => {
        $("sidebar").classList.toggle("open");
      });
    }

    if ($("refreshOverview")) {
      $("refreshOverview").addEventListener("click", () => {
        loadOverview().catch(console.error);
      });
    }

    if ($("loadAnalytics")) {
      $("loadAnalytics").addEventListener("click", () => {
        loadAnalytics().catch(console.error);
      });
    }

    if ($("realtimeForm")) {
      $("realtimeForm").addEventListener("submit", runRealtime);
    }

    if ($("downloadMonthlyReport")) {
      $("downloadMonthlyReport").addEventListener("click", () => {
        window.location.href = "/api/download/monthly-report";
      });
    }

    if ($("downloadCycleReport")) {
      $("downloadCycleReport").addEventListener("click", () => {
        const cycleStart = $("cycleStart") ? $("cycleStart").value : "";
        window.location.href = `/api/download/cycle-report?cycle_start=${cycleStart}`;
      });
    }

    setupApplianceInputs();
  }

  // =====================================================
  // Solar Gold cursor
  // =====================================================

  function setupSolarGoldCursor() {
    const supportsFinePointer = window.matchMedia("(pointer: fine)").matches;
    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (!supportsFinePointer || prefersReducedMotion) {
      return;
    }

    document.querySelectorAll(".solar-cursor-dot, .solar-cursor-ring").forEach((el) => {
      el.remove();
    });

    const dot = document.createElement("div");
    const ring = document.createElement("div");

    dot.className = "solar-cursor-dot";
    ring.className = "solar-cursor-ring";

    document.body.appendChild(ring);
    document.body.appendChild(dot);
    document.body.classList.add("solar-cursor-enabled");

    let mouseX = window.innerWidth / 2;
    let mouseY = window.innerHeight / 2;

    let ringX = mouseX;
    let ringY = mouseY;

    let ringScale = 1;
    let targetScale = 1;

    const interactiveSelector = [
      "a",
      "button",
      "input",
      "select",
      "textarea",
      "[role='button']",
      ".nav-item",
      ".primary-btn",
      ".kpi-card",
      ".panel",
      ".quick-card",
      ".hero-card",
      ".theme-toggle-btn",
    ].join(",");

    function setDotPosition(x, y) {
      dot.style.transform = `translate3d(${x}px, ${y}px, 0) translate(-50%, -50%)`;
    }

    function setRingPosition(x, y, scale) {
      ring.style.transform = `translate3d(${x}px, ${y}px, 0) translate(-50%, -50%) scale(${scale})`;
    }

    window.addEventListener(
      "mousemove",
      (event) => {
        mouseX = event.clientX;
        mouseY = event.clientY;

        document.body.classList.add("solar-cursor-visible");

        setDotPosition(mouseX, mouseY);
      },
      { passive: true }
    );

    window.addEventListener("mouseenter", () => {
      document.body.classList.add("solar-cursor-visible");
    });

    window.addEventListener("mouseleave", () => {
      document.body.classList.remove("solar-cursor-visible");
    });

    document.addEventListener(
      "mouseover",
      (event) => {
        if (event.target.closest(interactiveSelector)) {
          document.body.classList.add("solar-cursor-hover");
          targetScale = 1.5;
        }
      },
      { passive: true }
    );

    document.addEventListener(
      "mouseout",
      (event) => {
        if (event.target.closest(interactiveSelector)) {
          document.body.classList.remove("solar-cursor-hover");
          targetScale = 1;
        }
      },
      { passive: true }
    );

    window.addEventListener("mousedown", () => {
      document.body.classList.add("solar-cursor-active");
      targetScale = 0.62;
    });

    window.addEventListener("mouseup", () => {
      document.body.classList.remove("solar-cursor-active");

      if (document.body.classList.contains("solar-cursor-hover")) {
        targetScale = 1.5;
      } else {
        targetScale = 1;
      }
    });

    function animateRing() {
      ringX += (mouseX - ringX) * 0.22;
      ringY += (mouseY - ringY) * 0.22;
      ringScale += (targetScale - ringScale) * 0.22;

      setRingPosition(ringX, ringY, ringScale);

      requestAnimationFrame(animateRing);
    }

    animateRing();
  }

  // =====================================================
  // Theme toggle
  // =====================================================

  function setupThemeToggle() {
    function applyTheme(theme) {
      document.body.classList.remove("theme-light", "theme-dark");

      if (theme === "dark") {
        document.body.classList.add("theme-dark");
      } else {
        document.body.classList.add("theme-light");
      }

      localStorage.setItem("tneb-theme", theme);

      const icon = $("themeToggleIcon");

      if (icon) {
        icon.textContent = theme === "dark" ? "☀️" : "🌙";
      }
    }

    const savedTheme = localStorage.getItem("tneb-theme") || "light";

    applyTheme(savedTheme);

    const button = $("themeToggle");

    if (button) {
      button.addEventListener("click", () => {
        const isDark = document.body.classList.contains("theme-dark");
        applyTheme(isDark ? "light" : "dark");
      });
    }
  }

  // =====================================================
  // App init
  // =====================================================

  async function init() {
    setupThemeToggle();
    setupEvents();
    setupSolarGoldCursor();

    await loadMeta();
    await loadOverview();
  }

  document.addEventListener("DOMContentLoaded", () => {
    init().catch((error) => {
      console.error(error);
      alert(`App error: ${error.message}`);
    });
  });
})();