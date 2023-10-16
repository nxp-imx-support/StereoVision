/*
 * Copyright 2023 NXP
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "gpiod.h"

int main(int argc, char *argv[]) {

  struct gpiod_chip *chip;
  struct gpiod_line *line;
  char *dev;
  int gpio_line;
  int i, req, value = 0;

  if (argc != 3) {
    printf("Wrong parameters \n \
        Usage:\n \
        ./gpio_toggle_example <gpio_dev> <line>\n \
        Example:\n \
        ./gpio_toggle_example /dev/gpiochip4 9 \n");

    return -1;
  }

  dev = strdup(argv[1]);

  if (!dev) {
    printf("Wrong CharDev Name\n");
    return -1;
  }

  if (argv[2]) {
    gpio_line = atoi(strdup(argv[2]));
  }

  chip = gpiod_chip_open(dev);

  if (!chip) {
    return -1;
  }

  line = gpiod_chip_get_line(chip, gpio_line);

  if (!line) {
    gpiod_chip_close(chip);
    return -1;
  }

  /* config GPIO pin as output and its description */
  req = gpiod_line_request_output(line, "button", GPIOD_LINE_ACTIVE_STATE_LOW);

  if (req) {
    gpiod_chip_close(chip);
    return -1;
  }

  while (true) {

    if (gpiod_line_set_value(line, value) != 0) {

      printf("Impossible to change line %d value to %d \n", gpio_line, value);

      gpiod_chip_close(chip);

      return -1;
    }

    value ^= 1;
    usleep(120000);
  }

  /* release the line for input usage */
  gpiod_line_release(line);

  gpiod_chip_close(chip);
}
