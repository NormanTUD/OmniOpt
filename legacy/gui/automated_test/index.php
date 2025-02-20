<title>OmniOpt-automated test</title>
<?php
	if(file_exists("log.csv")) {
		$contents = file_get_contents("log.csv");

		$array = preg_split("/\r\n|\n|\r/", $contents);
		print "<table border='1'>\n";
		print "<tr><th>Time</th><th>Failed tests</th></tr>\n";
		$array = array_reverse($array);
		foreach ($array as $line) {
			$line_split = preg_split("/;/", $line);
			if(count($line_split) >= 2) {
				print "\t<tr>\n";
				print "\t\t<td>\n";
				print "\t\t\t".$line_split[0]."\n";
				print "\t\t</td>\n";
				print "\t\t<td>\n";
				print "\t\t\t<pre>\n";
				for ($i = 1; $i <= count($line_split); $i++) {
					print $line_split[$i];
					if($i != count($line_split)) {
						print "\n";
					}
				}
				print "</pre>\n";
				print "\t\t</td>\n";
				print "\t</tr>\n";
			}
		}
		print "</table>";
	} else {
		print "log.csv does not exist";
	}
?>
