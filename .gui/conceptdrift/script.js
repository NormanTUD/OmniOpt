$(document).ready(function () {
  $('.header-section').click(function(){
    $('.header-section').removeClass("active");
    $(this).addClass("active");
  });

  // Function to fade out landing title with animation
  function hideLandingTitle() {
    const $landingTitle = $('#landing-title');
    $landingTitle.addClass('fade-out');
    setTimeout(() => {
      $landingTitle.addClass('hidden');
    }, 600); // Wait for fade animation to complete
  }

  // Function to show landing title with fade in
  function showLandingTitle() {
    const $landingTitle = $('#landing-title');
    $landingTitle.removeClass('hidden').removeClass('fade-out');
  }

  // Handle click on "Home"
  $('#home').click(function () {
    // Show landing title and hide all content sections
    showLandingTitle();
    $('#single-dds-content').addClass('hidden');
    $('#pareto-fronts-content').addClass('hidden');
    $('#ensemble-estimation-content').addClass('hidden');
    $('#content').empty();
  });

  // Handle click on "Single DDs"
  $('#single-dds').click(function () {
    // Hide landing title with fade and show the Single DDs content
    hideLandingTitle();
    $('#pareto-fronts-content').addClass('hidden');
    $('#ensemble-estimation-content').addClass('hidden');
    $('#single-dds-content').removeClass('hidden');
    $('#content').empty();
  });

  // Handle click on the "Show" button
  $('#show-button_sdd').click(function () {
    // Check if the selected metric is ACCURACY-RUNTIME-REQLABELS
    const metrics = $('custom-dropdown[name="metrics"] input[type="hidden"]').val();
    const basePath = (metrics === 'ACCURACY-RUNTIME-REQLABELS') ? 'exps_acc_run_req/' : 'exps/';
    handleShowButton(this, ['dd', 'dataset', 'model', 'metrics'], basePath);
  });

  $('#show-button_par').click(function () {
    handleShowButton(this, ['dataset_par', 'model_par', 'metrics_par'], 'paretos/', '.html', '_');
  });

  // Handle click on "Pareto Fronts"
  $('#pareto-fronts').click(function () {
    hideLandingTitle();
    $('#pareto-fronts-content').removeClass('hidden');
    $('#single-dds-content').addClass('hidden');
    $('#ensemble-estimation-content').addClass('hidden');
    $('#content').empty();
  });

  // Handle click on "Ensemble Estimation"
  $('#ensemble-estimation').click(function () {
    hideLandingTitle();
    $('#ensemble-estimation-content').removeClass('hidden');
    $('#single-dds-content').addClass('hidden');
    $('#pareto-fronts-content').addClass('hidden');
    $('#content').empty();
  });

  $('#box1').click(function() {
    $('#menu1').toggleClass('hidden');
    $('#menu2').removeClass('menu-visible');
  });

  $('#box2').click(function(e) {
    e.preventDefault();
    $('#menu1').addClass('hidden');
    
    // Show enhanced loading state
    const $content = $('#content');
    $content.html(`
      <div class="ensemble-loading">
        <div class="loading-spinner"></div>
        <h3>Loading Single Variate DDs Visualization</h3>
        <p>Preparing multiple features visualization...</p>
      </div>
    `);
    $content.css('opacity', '1');
    
    // Create an enhanced iframe container
    const iframeContainer = document.createElement('div');
    iframeContainer.className = 'ensemble-iframe-container';
    
    // Create the iframe with enhanced styling
    const iframe = document.createElement('iframe');
    iframe.src = `sv_mf/sv_mf.html`;
    iframe.className = 'ensemble-iframe';
    iframe.style.width = '100%';
    iframe.style.height = '2000px';
    iframe.style.minHeight = '1500px';
    iframe.style.border = 'none';
    iframe.style.borderRadius = '12px';
    iframe.style.boxShadow = '0 8px 32px rgba(2, 45, 63, 0.15)';
    iframe.style.background = 'white';
    iframe.style.transition = 'all 0.3s ease';
    iframe.style.opacity = '0';
    iframe.style.transform = 'translateY(20px)';
    
    // Add iframe to container
    iframeContainer.appendChild(iframe);
    
    // Add loading animation and smooth transition
    iframe.onload = function() {
      // Fade out loading, fade in iframe
      setTimeout(() => {
        $('.ensemble-loading').fadeOut(300, () => {
          iframe.style.opacity = '1';
          iframe.style.transform = 'translateY(0)';
        });
      }, 500);
    };
    
    // Clear the content and append the iframe container
    setTimeout(() => {
      $content.append(iframeContainer);
    }, 100);
    
    // Enhanced error handling
    iframe.onerror = function() {
      $content.html(`
        <div class="ensemble-error">
          <div class="error-icon">⚠️</div>
          <h3>Visualization Loading Error</h3>
          <p>Unable to load the Single Variate DDs visualization.</p>
          <p>Please try refreshing the page.</p>
          <button onclick="location.reload()" class="retry-button">Retry</button>
        </div>
      `);
    };
  });


  $('#show-button_ee_ensemble').click(function (e) {
    e.preventDefault();
    
    // Get the selected dataset
    const dataset = $('custom-dropdown[name="dataset_ee"] input[type="hidden"]').val();
    
    if (!dataset) {
      console.error('Please select a dataset');
      return;
    }
    
    // Show enhanced loading state
    const $content = $('#content');
    $content.html(`
      <div class="ensemble-loading">
        <div class="loading-spinner"></div>
        <h3>Loading Ensemble Visualization</h3>
        <p>Preparing ${dataset} dataset visualization...</p>
      </div>
    `);
    $content.css('opacity', '1');
    
    // Create an enhanced iframe container
    const iframeContainer = document.createElement('div');
    iframeContainer.className = 'ensemble-iframe-container';
    
    // Create the iframe with enhanced styling
    const iframe = document.createElement('iframe');
    iframe.src = `ensemble/ensemble.html?dataset=${encodeURIComponent(dataset)}`;
    iframe.className = 'ensemble-iframe';
    iframe.style.width = '100%';
    iframe.style.height = '1200px';
    iframe.style.minHeight = '800px';
    iframe.style.border = 'none';
    iframe.style.borderRadius = '12px';
    iframe.style.boxShadow = '0 8px 32px rgba(2, 45, 63, 0.15)';
    iframe.style.background = 'white';
    iframe.style.transition = 'all 0.3s ease';
    iframe.style.opacity = '0';
    iframe.style.transform = 'translateY(20px)';
    
    // Add iframe to container
    iframeContainer.appendChild(iframe);
    
    // Add loading animation and smooth transition
    iframe.onload = function() {
      // Fade out loading, fade in iframe
      setTimeout(() => {
        $('.ensemble-loading').fadeOut(300, () => {
          iframe.style.opacity = '1';
          iframe.style.transform = 'translateY(0)';
        });
      }, 500);
    };
    
    // Clear the content and append the iframe container
    setTimeout(() => {
      $content.append(iframeContainer);
    }, 100);
    
    // Enhanced error handling
    iframe.onerror = function() {
      $content.html(`
        <div class="ensemble-error">
          <div class="error-icon">⚠️</div>
          <h3>Visualization Loading Error</h3>
          <p>Unable to load the ensemble visualization for <strong>${dataset}</strong>.</p>
          <p>Please try selecting a different dataset or refresh the page.</p>
          <button onclick="location.reload()" class="retry-button">Retry</button>
        </div>
      `);
    };
  });

  // Download Modal Button Color Override
  // Monitor for dynamically created download modals and fix button colors
  const observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
      mutation.addedNodes.forEach(function(node) {
        if (node.nodeType === 1 && node.style && 
            node.style.position === 'fixed' && 
            node.style.zIndex === '10000') {
          
          // Wait a bit for the modal content to be fully created
          setTimeout(function() {
            const buttons = node.querySelectorAll('button');
            buttons.forEach(function(button) {
              const buttonText = button.textContent || button.innerHTML;
              
              // Apply custom colors based on button content
              if (buttonText.includes('Download Current Experiment')) {
                button.style.backgroundColor = '#015E80';
                button.style.background = '#015E80';
                
                // Add hover effects
                button.addEventListener('mouseenter', function() {
                  this.style.backgroundColor = '#016A8E';
                });
                button.addEventListener('mouseleave', function() {
                  this.style.backgroundColor = '#015E80';
                });
                
              } else if (buttonText.includes('Download All Data')) {
                button.style.backgroundColor = '#047A9A';
                button.style.background = '#047A9A';
                
                // Add hover effects
                button.addEventListener('mouseenter', function() {
                  this.style.backgroundColor = '#016A8E';
                });
                button.addEventListener('mouseleave', function() {
                  this.style.backgroundColor = '#047A9A';
                });
                
              } else if (buttonText.includes('Cancel')) {
                button.style.backgroundColor = '#024A65';
                button.style.background = '#024A65';
                
                // Add hover effects
                button.addEventListener('mouseenter', function() {
                  this.style.backgroundColor = '#002D3F';
                });
                button.addEventListener('mouseleave', function() {
                  this.style.backgroundColor = '#024A65';
                });
              }
            });
          }, 50);
        }
      });
    });
  });
  
  // Start observing for modal creation
  observer.observe(document.body, { childList: true, subtree: true });
});
