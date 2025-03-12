<?php
	http_response_code(404);

	include("_header_base.php");
?>

<h2>Error: Not found</h2>

<?php
	include("footer.php");
	exit(1);
?>
