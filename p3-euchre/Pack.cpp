// Project UID 1d9f47bfc76643019cfbf037641defe1
#include <cassert>
#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <string>
#include <array>
#include <vector>
#include <sstream>
#include <algorithm>
#include "Pack.h"

Pack::Pack() {
    int pack_pos = 0;
    for (int suit = 0; suit < 4; suit++) {
        for (int rank = 7; rank < 13; rank++) {
            cards[pack_pos] = Card(RANK_NAMES_BY_WEIGHT[rank],
                                   SUIT_NAMES_BY_WEIGHT[suit]);
            pack_pos++;
        }
    }
    next = 0;
}

Pack::Pack(std::istream& pack_input) {
    int pack_pos = 0;
    while (pack_pos < PACK_SIZE) {
        std::string rank = "";
        std::string suit = "";
        std::string of = "";
        pack_input >> rank;
        pack_input >> of;
        pack_input >> suit;
        cards[pack_pos] = Card(rank, suit);
        pack_pos++;
    }
    next = 0;
}

Card Pack::deal_one() {
    if (empty()) {
        reset();
    }
    int temp = next;
    ++next;
    return cards[temp];
}

void Pack::reset() {
    next = 0;
    return;
}

void Pack::shuffle() {
    for (int l = 0; l < 7; ++l) {
        std::array<Card, PACK_SIZE> cards_duplicate;
        for (int i = 0; i < PACK_SIZE; i++) {
            cards_duplicate[i] = Card(cards[i].get_rank(), cards[i].get_suit());
        }
        for (int j = 0; j < PACK_SIZE / 2; ++j) {
            cards[2 * j + 1] = Card(cards_duplicate[j].get_rank(),
                                cards_duplicate[j].get_suit());
        }
        for (int k = 0; k < PACK_SIZE / 2; ++k) {
            cards[2 * k] =
            Card(cards_duplicate[PACK_SIZE / 2 + k].get_rank(),
                cards_duplicate[PACK_SIZE / 2 + k].get_suit());
        }
    }
    reset();
    return;
}

bool Pack::empty() const {
    if (next == PACK_SIZE) {
        return true;
    }
    return false;
}






