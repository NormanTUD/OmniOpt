#!/usr/bin/perl

use strict;
use warnings;

my $x = shift @ARGV // 10000000000000;
my $y = shift @ARGV // 10000000000000;
my $param = $x * $y;

print "HALLO: ".rand()."\n";
print "WELT: ".rand()."\n";
print "DAS: ".rand()."\n";
print "IST: ".rand()."\n";
print "EIN: ".rand()."\n";
print "TEST: ".rand()."\n";
print "RESULT: $param\n";
