class CustomDropdown extends HTMLElement {
  connectedCallback() {
    const label = this.getAttribute('label');
    const name = this.getAttribute('name');
    const defaultValue = this.getAttribute('default-value') || this.getAttribute('value') || '';
    const useExternalLabel = this.hasAttribute('external-label');
    const options = JSON.parse(this.getAttribute('options') || '[]');
    
    // Find the default option text and value
    let defaultText = '';
    let defaultOptionValue = '';
    
    // Process options to handle string, object, and optgroup formats
    let processedOptions = [];
    
    options.forEach(opt => {
      if (typeof opt === 'string') {
        processedOptions.push({ label: opt, value: opt });
      } else if (opt.optgroup) {
        // Handle optgroup structure
        opt.options.forEach(groupOpt => {
          if (typeof groupOpt === 'string') {
            processedOptions.push({ 
              label: groupOpt, 
              value: groupOpt, 
              optgroup: opt.optgroup 
            });
          } else {
            processedOptions.push({
              label: groupOpt.label || groupOpt.value || '',
              value: groupOpt.value || groupOpt.label || '',
              optgroup: opt.optgroup
            });
          }
        });
      } else {
        processedOptions.push({
          label: opt.label || opt.value || '',
          value: opt.value || opt.label || ''
        });
      }
    });
    
    // Find the default option
    if (defaultValue) {
      const defaultOption = processedOptions.find(opt => 
        opt.value === defaultValue || opt.label === defaultValue
      ) || processedOptions[0];
      
      if (defaultOption) {
        defaultText = defaultOption.label;
        defaultOptionValue = defaultOption.value;
      }
    } else if (processedOptions.length > 0) {
      defaultText = processedOptions[0].label;
      defaultOptionValue = processedOptions[0].value;
    }

    // Create the dropdown HTML
    this.innerHTML = `
      <div class="dropdown-container">
        <div class="dropdown-header">
          <span class="dropdown-label">${useExternalLabel ? defaultText : `${label}${defaultText ? `: ${defaultText}` : ''}`}</span>
          <span class="dropdown-arrow">&#x203A;</span>
        </div>
        <div class="dropdown-content">
          <svg class="svg-arrow" width="100%" height="9">
            <path d="M0,9 L15,9 M23,0 M31,9 L300,9" fill="white" stroke="black" stroke-width="1.5" fill-opacity="0" />
            <path d="M15,9 L23,0 L31,9" fill="white" stroke="black" stroke-width="1" fill-opacity="0" />
          </svg>
          ${(() => {
            let html = '';
            let currentOptgroup = null;
            
            processedOptions.forEach(opt => {
              // Add optgroup header if needed
              if (opt.optgroup && opt.optgroup !== currentOptgroup) {
                if (currentOptgroup !== null) {
                  html += '</div>'; // Close previous optgroup
                }
                html += `<div class="optgroup"><div class="optgroup-label">${opt.optgroup}</div>`;
                currentOptgroup = opt.optgroup;
              } else if (!opt.optgroup && currentOptgroup !== null) {
                html += '</div>'; // Close optgroup if moving to non-grouped options
                currentOptgroup = null;
              }
              
              const isSelected = opt.value === defaultOptionValue;
              html += `<p data-value="${opt.value}" ${isSelected ? 'class="selected"' : ''}>${opt.label}</p>`;
            });
            
            if (currentOptgroup !== null) {
              html += '</div>'; // Close final optgroup
            }
            
            return html;
          })()}
        </div>
        <input type="hidden" name="${name}" value="${defaultOptionValue}" id="${name}-input">
      </div>
    `;

    const container = this.querySelector('.dropdown-container');
    this.initDropdown(container);
  }

  initDropdown(container) {
    const $container = $(container);
    const $header = $container.find('.dropdown-header');
    const $label = $container.find('.dropdown-label');
    const $content = $container.find('.dropdown-content');
    const $input = $container.find('input[type="hidden"]');
    const useExternalLabel = this.hasAttribute('external-label');
    const baseText = useExternalLabel ? '' : $label.text().split(':')[0].trim();

    $label.data('base', baseText);

    function closeAllDropdowns() {
      $('.dropdown-container').removeClass('dropdown-open');
      $('.single-dd-field').removeClass('dropdown-active');
      $('.pareto-field').removeClass('dropdown-active');
      $('.dropdown-header').removeClass('open');
      $('.dropdown-content').removeClass('show');
    }

    $header.on('click', function(e) {
      e.stopPropagation();

      const shouldOpen = !$content.hasClass('show');
      closeAllDropdowns();

      if (shouldOpen) {
        $container.addClass('dropdown-open');
        $container.closest('.single-dd-field, .pareto-field').addClass('dropdown-active');
        $header.addClass('open');
        $content.addClass('show');
      }
    });

    $content.on('click', 'p', function(e) {
      e.stopPropagation();

      const $option = $(this);
      const value = $option.data('value');
      const text = $option.text();

      $content.find('p').removeClass('selected');
      $option.addClass('selected');

      $label.text(useExternalLabel ? text : `${baseText}: ${text}`);
      $input.val(value).trigger('change');

      $container.removeClass('dropdown-open');
      $container.closest('.single-dd-field, .pareto-field').removeClass('dropdown-active');
      $header.removeClass('open');
      $content.removeClass('show');
    });

    if (!window.__customDropdownOutsideClickBound) {
      $(document).on('click', function(e) {
        if ($(e.target).closest('.dropdown-container').length === 0) {
          closeAllDropdowns();
        }
      });
      window.__customDropdownOutsideClickBound = true;
    }
  }
}

customElements.define('custom-dropdown', CustomDropdown);
