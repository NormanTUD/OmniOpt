<?php
	if(!function_exists("dier")) {
		function dier ($data, $enable_html = 0, $exception = 0) {
			$source_data = debug_backtrace()[0];
			@$source = 'Aufgerufen von <b>'.debug_backtrace()[1]['file'].'</b>::<i>'.debug_backtrace()[1]['function'].'</i>, line '.htmlentities($source_data['line'])."<br>\n";
			$print = $source;

			$print .= "<pre>\n";
			ob_start();
			print_r($data);
			$buffer = ob_get_clean();
			if($enable_html) {
				$print .= $buffer;
			} else {
				$print .= htmlentities($buffer);
			}
			$print .= "</pre>\n";

			$print .= "Backtrace:\n";
			$print .= "<pre>\n";
			foreach (debug_backtrace() as $trace) {
				$print .= htmlentities(sprintf("\n%s:%s %s", $trace['file'], $trace['line'], $trace['function']));
			}
			$print .= "</pre>\n";

			if(!$exception) {
				print $print;
				exit();
			} else {
				throw new Exception($print);
			}
		}
	}
?>
