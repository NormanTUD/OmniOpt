#!/usr/bin/perl

use strict;
use warnings;

my $param = shift @ARGV // 10000000000000;
$param *= 10;

print "HALLO: ".rand()."\n";
print "WELT: ".rand()."\n";
print "DAS: ".rand()."\n";
print "IST: ".rand()."\n";
print "EIN: ".rand()."\n";
print "TEST: ".rand()."\n";
print "RESULT: $param\n";
