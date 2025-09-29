$(document).ready(function () {
  $('.header-section').click(function(){
    $('.header-section').removeClass("active");
    $(this).addClass("active");
  });

  // Handle click on "Single DDs"
  $('#single-dds').click(function () {
    // Show the Single DDs content
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

  // Placeholder for other header options (Pareto Fronts, Ensemble Estimation)
  $('#pareto-fronts').click(function () {
    $('#pareto-fronts-content').removeClass('hidden');
    $('#single-dds-content').addClass('hidden');
    $('#ensemble-estimation-content').addClass('hidden');
    $('#content').empty();
  });

  $('#ensemble-estimation').click(function () {
    $('#ensemble-estimation-content').removeClass('hidden');
    $('#single-dds-content').addClass('hidden');
    $('#pareto-fronts-content').addClass('hidden');
    $('#content').empty();
  });

  $('#box1').click(function() {
        $('#menu1').toggle();
        $('#menu2').hide();
      });

  $('#box2').click(function() {
    $('#menu1').hide();
    $('#menu2').toggle();
  });


  $('#show-button_ee_ensemble').click(function (e) {
    e.preventDefault();
    
    // Get the selected dataset
    const dataset = $('custom-dropdown[name="dataset_ee"] input[type="hidden"]').val();
    
    if (!dataset) {
      console.error('Please select a dataset');
      return;
    }
    
    // Show loading state
    const $content = $('#content');
    $content.html('<div class="loading">Loading ensemble visualization...</div>');
    $content.css('opacity', '1');
    
    // Create an iframe to load ensemble.html with the dataset parameter
    const iframe = document.createElement('iframe');
    iframe.src = `ensemble/ensemble.html?dataset=${encodeURIComponent(dataset)}`;
    iframe.style.width = '100%';
    iframe.style.height = '1200px';  // Increased height
    iframe.style.minHeight = '800px';  // Minimum height
    iframe.style.border = '1px solid #e0e0e0';
    iframe.style.borderRadius = '4px';
    iframe.style.marginTop = '10px';
    
    // Clear the content and append the iframe
    $content.empty().append(iframe);
    
    // Handle errors
    iframe.onerror = function() {
      $content.html(`
        <div class="error">
          <h3>Error loading visualization</h3>
          <p>Failed to load the ensemble visualization.</p>
        </div>
      `);
    };
  });
});

