#!/usr/bin/perl

use strict;
use warnings;
use Data::Dumper;
use POSIX;

my $date = strftime "%F %T", localtime time;

my @failed_tests = map { chomp; $_ } ($date, qx(ssh -X -o StrictHostKeyChecking=no scads\@taurus.hrsk.tu-dresden.de -t "curl https://imageseg.scads.ai/omnioptgui/autorun_test.sh | bash -l -" 2>/dev/null | grep 'FAILED TEST' | sed -e 's/.* -> //' | sed -r "s/\\x1B\\[([0-9]{1,3}(;[0-9]{1,2})?)?[mGK]//g" | sed -e 's/\\r//g'));


if(@failed_tests == 0) {
	push @failed_tests, "OK";
}

print join(";", @failed_tests);

open my $fh, '>>', '/var/www/html/omnioptgui/automated_test/log.csv';
print $fh join(";", @failed_tests)."\n";
close $fh;
