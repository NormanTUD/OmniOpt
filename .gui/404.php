<?php
	include_once("_functions.php");

	$prefix = isset($_SERVER["CONTEXT_PREFIX"]) ? $_SERVER["CONTEXT_PREFIX"] : "";
	$requested_file = $_SERVER["REQUEST_URI"];

	$escaped_prefix = preg_quote($prefix, '/');
	$pattern = '/^' . $escaped_prefix . '/';

	$filename = preg_replace($pattern, '', $requested_file);

	if($filename == "install_omniax.sh" || $filename == "install_omniopt.sh" || $filename == "install_omniopt2.sh") {
		include("install_omniax.php");
		exit(0);
	}

	http_response_code(404);

	include("_header_base.php");
?>

<h2>Error: Not found</h2>

<?php
	include("footer.php");
	exit(1);
?>
