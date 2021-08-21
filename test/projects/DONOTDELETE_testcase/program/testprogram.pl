#!/bin/perl

use strict;
use warnings;

my $arg1 = shift @ARGV;
my $arg2 = shift @ARGV;
my $arg3 = shift @ARGV;
my $arg4 = shift @ARGV;

my $summiert = ($arg1 + $arg2 + $arg3 + $arg4);

print "summiert: ".$summiert."\n";
print "loss: 0.5\n";
print "testq: 355\n";
print "RESULT: ".$summiert."\n";
