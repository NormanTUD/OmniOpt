#!/usr/bin/perl

use Time::HiRes qw/sleep/;
use strict;

for (0 .. 10000) {
    print (("xx" x 80)."\n");
    sleep 0.0005;
}
