function initDropdown(containerSelector) {
  const $container = $(containerSelector);
  const $header = $container.find('.dropdown-header');
  const $label = $container.find('.dropdown-label');
  const $content = $container.find('.dropdown-content');
  const $input = $container.find('input[type="hidden"]');
  const initialValue = $input.val();

  // Set initial value if it exists
  if (initialValue) {
    const $selectedItem = $content.find(`p[data-value="${initialValue}"]`);
    if ($selectedItem.length) {
      $label.text(`${$label.data('base')}: ${$selectedItem.text()}`);
      $selectedItem.addClass('selected');
    }
  }

  $header.on('click', function () {
    $header.toggleClass('open');
    $content.toggleClass('show');
  });

  $content.find('p').on('click', function () {
    const $p = $(this);
    const text = $p.text();
    const value = $p.data('value') ?? text;

    // Update UI
    $content.find('p').removeClass('selected');
    $p.addClass('selected');
    $label.text(`${$label.data('base')}: ${text}`);
    $input.val(value);
    
    // Close dropdown
    $content.removeClass('show');
    $header.removeClass('open');
  });

  // Close when click outside of dropdown
  $(document).on('click', function (e) {
    if (!$container.is(e.target) && $container.has(e.target).length === 0) {
      $content.removeClass('show');
      $header.removeClass('open');
    }
  });

  // Preserve label base name (before any choice is made)
  const baseText = $label.text().split(':')[0].trim();
  $label.data('base', baseText);
}

class CustomDropdown extends HTMLElement {
  connectedCallback() {
    const label = this.getAttribute('label');
    const name = this.getAttribute('name');
    const defaultValue = this.getAttribute('default-value') || this.getAttribute('value') || '';
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
          <span class="dropdown-label">${label}${defaultText ? `: ${defaultText}` : ''}</span>
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
    
    // Initialize the dropdown
    const container = this.querySelector('.dropdown-container');
    this.initDropdown(container);
  }
  
  initDropdown(container) {
    const $container = $(container);
    const $header = $container.find('.dropdown-header');
    const $label = $container.find('.dropdown-label');
    const $content = $container.find('.dropdown-content');
    const $input = $container.find('input[type="hidden"]');
    const baseText = $label.text().split(':')[0].trim();
    
    // Store the base label text
    $label.data('base', baseText);
    
    // Toggle dropdown
    $header.on('click', function() {
      $header.toggleClass('open');
      $content.toggleClass('show');
    });
    
    // Handle option selection
    $content.on('click', 'p', function() {
      const $option = $(this);
      const value = $option.data('value');
      const text = $option.text();
      
      // Update UI
      $content.find('p').removeClass('selected');
      $option.addClass('selected');
      $label.text(`${baseText}: ${text}`);
      
      // Update hidden input
      $input.val(value);
      
      // Close dropdown
      $header.removeClass('open');
      $content.removeClass('show');
    });
    
    // Close when clicking outside
    $(document).on('click', (e) => {
      if (!$container.is(e.target) && $container.has(e.target).length === 0) {
        $header.removeClass('open');
        $content.removeClass('show');
      }
    });
  }
}

customElements.define('custom-dropdown', CustomDropdown);