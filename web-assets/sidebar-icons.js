(() => {
  function setupSidebars() {
    const layout = document.querySelector(".md-main__inner");
    const title = document.querySelector(".md-header__title");
    if (!layout || !title) return;

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
      button.setAttribute("aria-controls", sidebar.id);
      const icon = document.createElement("span");
      icon.className = "wiki-sidebar-icon";
      icon.setAttribute("aria-hidden", "true");
      button.append(icon);
      controls.append(button);

      function updateLayout() {
        sidebar.hidden = desktop.matches && collapsed;
        const hint = `${collapsed ? "展开" : "收起"}${label}`;
        button.dataset.tooltip = hint;
        button.setAttribute("aria-label", hint);
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

    if (controls.childElementCount) title.after(controls);
  }

  setupSidebars();
})();
