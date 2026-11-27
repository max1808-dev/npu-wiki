(() => {
  function setupSidebars() {
    const layout = document.querySelector(".md-main__inner");
    const article = layout?.querySelector(".md-content__inner");
    if (!article || article.querySelector(".wiki-sidebar-controls")) return;

    const controls = document.createElement("div");
    controls.className = "wiki-sidebar-controls";
    controls.setAttribute("role", "group");
    controls.setAttribute("aria-label", "阅读布局");

    const sidebars = [
      { type: "primary", label: "左侧导航", breakpoint: "76.25em" },
      { type: "secondary", label: "右侧目录", breakpoint: "60em" },
    ];

    for (const { type, label, breakpoint } of sidebars) {
      const sidebar = layout.querySelector(`.md-sidebar--${type}`);
      if (!sidebar || sidebar.hidden) continue;

      const storageKey = `npu-wiki-sidebar-${type}`;
      const desktop = window.matchMedia(`(min-width: ${breakpoint})`);
      let collapsed = false;
      try {
        collapsed = localStorage.getItem(storageKey) === "collapsed";
      } catch {
        // The controls still work when browser storage is unavailable.
      }

      sidebar.id = storageKey;
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.sidebar = type;
      button.textContent = `收起${label}`;
      button.setAttribute("aria-controls", sidebar.id);
      controls.append(button);

      function updateLayout() {
        sidebar.hidden = desktop.matches && collapsed;
        button.textContent = `${collapsed ? "展开" : "收起"}${label}`;
        button.setAttribute("aria-expanded", String(!sidebar.hidden));
      }

      button.addEventListener("click", () => {
        collapsed = !collapsed;
        updateLayout();
        try {
          localStorage.setItem(storageKey, collapsed ? "collapsed" : "expanded");
        } catch {
          // Keep the current page usable without storage.
        }
      });

      desktop.addEventListener("change", updateLayout);
      updateLayout();
    }

    if (controls.childElementCount) article.prepend(controls);
  }

  setupSidebars();
})();
