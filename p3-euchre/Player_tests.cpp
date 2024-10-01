// Project UID 1d9f47bfc76643019cfbf037641defe1

#include "Player.h"
#include "unit_test_framework.h"

#include <iostream>

using namespace std;

TEST(test_player_insertion) {
  // Create a Human player
  Player * human = Player_factory("NotRobot", "Human");

  // Print the player using the stream insertion operator
  ostringstream oss1;
  oss1 << * human;

  // Verify that the output is the player's name
  ASSERT_EQUAL(oss1.str(), "NotRobot");

  // Create a Simple player
  Player * alice = Player_factory("Alice", "Simple");

  // Print the player using the stream insertion operator
  ostringstream oss2;
  oss2 << *alice;
  ASSERT_EQUAL(oss2.str(), "Alice");

  // Clean up players that were created using Player_factory()
  delete human;
  delete alice;
}

TEST(test_simple_player_get_name) {
  // Create a player and verify that get_name() returns the player's name
  Player * alice = Player_factory("Alice", "Simple");
  ASSERT_EQUAL(alice->get_name(), "Alice");
  delete alice;
}

TEST(test_simple_player_get_name1) {
    Player * bob = Player_factory("Bob", "Simple");
    ASSERT_EQUAL("Bob", bob->get_name());

    delete bob;
}

TEST(test_simple_player_make_trump) {
  // Bob's hand
  Player * bob = Player_factory("Bob", "Simple");
  bob->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_TEN, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_QUEEN, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_KING, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_ACE, Card::SUIT_SPADES));

  // Bob makes tump
  Card nine_spades(Card::RANK_NINE, Card::SUIT_SPADES);
  string trump;
  bool orderup = bob->make_trump(
    nine_spades,    // Upcard
    true,           // Bob is also the dealer
    1,              // First round
    trump           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_TRUE(orderup);
  ASSERT_EQUAL(trump, Card::SUIT_SPADES);

  delete bob;
}

TEST(test_simple_player_make_trump1) {
  // Bob's hand
  Player * alice = Player_factory("Alice", "Simple");
  alice->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  alice->add_card(Card(Card::RANK_TEN, Card::SUIT_SPADES));
  alice->add_card(Card(Card::RANK_QUEEN, Card::SUIT_SPADES));
  alice->add_card(Card(Card::RANK_KING, Card::SUIT_SPADES));
  alice->add_card(Card(Card::RANK_ACE, Card::SUIT_SPADES));

  // Bob makes tump
  Card nine_spades(Card::RANK_NINE, Card::SUIT_SPADES);
  string trump;
  bool orderup = alice->make_trump(
    nine_spades,    // Upcard
    false,           // Bob is also the dealer
    1,              // First round
    trump           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_TRUE(orderup);
  ASSERT_EQUAL(trump, Card::SUIT_SPADES);

  delete alice;
}

TEST(test_simple_player_make_trump2) {
  Player * Tim = Player_factory("Tim", "Simple");
  Tim->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  Tim->add_card(Card(Card::RANK_TEN, Card::SUIT_SPADES));
  Tim->add_card(Card(Card::RANK_QUEEN, Card::SUIT_SPADES));
  Tim->add_card(Card(Card::RANK_KING, Card::SUIT_SPADES));
  Tim->add_card(Card(Card::RANK_ACE, Card::SUIT_SPADES));


  // Bob makes tump
  Card nine_hearts(Card::RANK_NINE, Card::SUIT_HEARTS);
  string trump;
  bool orderup = Tim->make_trump(
    nine_hearts,    // Upcard
    true,           // Bob is also the dealer
    1,              // First round
    trump           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_FALSE(orderup);

  delete Tim;
}

TEST(test_simple_player_make_trump3) {
  Player * Tom = Player_factory("Tom", "Simple");
  Tom->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  Tom->add_card(Card(Card::RANK_TEN, Card::SUIT_SPADES));
  Tom->add_card(Card(Card::RANK_QUEEN, Card::SUIT_SPADES));
  Tom->add_card(Card(Card::RANK_KING, Card::SUIT_SPADES));
  Tom->add_card(Card(Card::RANK_ACE, Card::SUIT_SPADES));


  // Bob makes tump
  Card nine_hearts(Card::RANK_NINE, Card::SUIT_HEARTS);
  string trump;
  bool orderup = Tom->make_trump(
    nine_hearts,    // Upcard
    true,           // Bob is also the dealer
    1,              // First round
    trump           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_FALSE(orderup);

  delete Tom;
}


TEST(test_simple_player_make_trump4) {
  Player * Tom = Player_factory("Tim", "Simple");
  Tom->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  Tom->add_card(Card(Card::RANK_TEN, Card::SUIT_SPADES));
  Tom->add_card(Card(Card::RANK_QUEEN, Card::SUIT_SPADES));
  Tom->add_card(Card(Card::RANK_KING, Card::SUIT_SPADES));
  Tom->add_card(Card(Card::RANK_ACE, Card::SUIT_SPADES));


  // Bob makes tump
  Card nine_hearts(Card::RANK_NINE, Card::SUIT_HEARTS);
  string trump;
  bool orderup = Tom->make_trump(
    nine_hearts,    // Upcard
    false,           // Bob is also the dealer
    1,              // First round
    trump           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_FALSE(orderup);

  delete Tom;
}

TEST(test_simple_player_make_trump5) {
  Player * Tim = Player_factory("Tim", "Simple");
  Tim->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  Tim->add_card(Card(Card::RANK_TEN, Card::SUIT_SPADES));
  Tim->add_card(Card(Card::RANK_KING, Card::SUIT_CLUBS));
  Tim->add_card(Card(Card::RANK_KING, Card::SUIT_HEARTS));
  Tim->add_card(Card(Card::RANK_ACE, Card::SUIT_SPADES));


  // Bob makes tump
  Card ten_spades(Card::RANK_TEN, Card::SUIT_SPADES);
  string trump;
  bool orderup = Tim->make_trump(
    ten_spades,    // Upcard
    true,           // Bob is also the dealer
    1,              // First round
    trump           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_FALSE(orderup);

  delete Tim;
}

TEST(test_simple_player_make_trump6) {
  Player * Tom = Player_factory("Tom", "Simple");
  Tom->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  Tom->add_card(Card(Card::RANK_TEN, Card::SUIT_SPADES));
  Tom->add_card(Card(Card::RANK_KING, Card::SUIT_CLUBS));
  Tom->add_card(Card(Card::RANK_KING, Card::SUIT_HEARTS));
  Tom->add_card(Card(Card::RANK_ACE, Card::SUIT_DIAMONDS));


  // Bob makes tump
  Card ten_spades(Card::RANK_TEN, Card::SUIT_SPADES);
  string trump;
  bool orderup = Tom->make_trump(
    ten_spades,    // Upcard
    false,           // Bob is also the dealer
    1,              // First round
    trump           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_FALSE(orderup);

  delete Tom;
}

TEST(test_simple_player_make_trump7) {
  Player * Tim = Player_factory("Tim", "Simple");
  Tim->add_card(Card(Card::RANK_ACE, Card::SUIT_SPADES));
  Tim->add_card(Card(Card::RANK_KING, Card::SUIT_SPADES));
  Tim->add_card(Card(Card::RANK_KING, Card::SUIT_CLUBS));
  Tim->add_card(Card(Card::RANK_KING, Card::SUIT_HEARTS));
  Tim->add_card(Card(Card::RANK_ACE, Card::SUIT_CLUBS));


  // Bob makes tump
  Card queen_spades(Card::RANK_QUEEN, Card::SUIT_SPADES);
  string trump;
  bool orderup = Tim->make_trump(
    queen_spades,    // Upcard
    true,           // Bob is also the dealer
    1,              // First round
    trump           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_TRUE(orderup);

  delete Tim;
}

TEST(test_simple_player_make_trump8) {
  Player * Tom = Player_factory("Tom", "Simple");
  Tom->add_card(Card(Card::RANK_QUEEN, Card::SUIT_SPADES));
  Tom->add_card(Card(Card::RANK_JACK, Card::SUIT_SPADES));
  Tom->add_card(Card(Card::RANK_KING, Card::SUIT_CLUBS));
  Tom->add_card(Card(Card::RANK_KING, Card::SUIT_HEARTS));
  Tom->add_card(Card(Card::RANK_ACE, Card::SUIT_CLUBS));


  // Bob makes tump
  Card queen_spades(Card::RANK_QUEEN, Card::SUIT_SPADES);
  string trump;
  bool orderup = Tom->make_trump(
    queen_spades,    // Upcard
    false,           // Bob is also the dealer
    1,              // First round
    trump           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_TRUE(orderup);

  delete Tom;
}

TEST(test_simple_player_make_trump9) {
  Player * Tim = Player_factory("Tim", "Simple");
  Tim->add_card(Card(Card::RANK_JACK, Card::SUIT_DIAMONDS));
  Tim->add_card(Card(Card::RANK_KING, Card::SUIT_DIAMONDS));
  Tim->add_card(Card(Card::RANK_TEN, Card::SUIT_CLUBS));
  Tim->add_card(Card(Card::RANK_KING, Card::SUIT_HEARTS));
  Tim->add_card(Card(Card::RANK_ACE, Card::SUIT_CLUBS));


  // Bob makes tump
  Card queen_diamonds(Card::RANK_QUEEN, Card::SUIT_DIAMONDS);
  string trump;
  bool orderup = Tim->make_trump(
    queen_diamonds,    // Upcard
    true,           // Bob is also the dealer
    1,              // First round
    trump           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_TRUE(orderup);

  delete Tim;
}

TEST(test_simple_player_make_trump10) {
  Player * Tom = Player_factory("Tim", "Simple");
  Tom->add_card(Card(Card::RANK_JACK, Card::SUIT_DIAMONDS));
  Tom->add_card(Card(Card::RANK_KING, Card::SUIT_DIAMONDS));
  Tom->add_card(Card(Card::RANK_TEN, Card::SUIT_CLUBS));
  Tom->add_card(Card(Card::RANK_KING, Card::SUIT_HEARTS));
  Tom->add_card(Card(Card::RANK_ACE, Card::SUIT_CLUBS));


  // Bob makes tump
  Card queen_diamonds(Card::RANK_QUEEN, Card::SUIT_DIAMONDS);
  string trump;
  bool orderup = Tom->make_trump(
    queen_diamonds,    // Upcard
    false,           // Bob is also the dealer
    1,              // First round
    trump           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_TRUE(orderup);

  delete Tom;
}

TEST(test_simple_player_make_trump11) {
  Player * Tim = Player_factory("Tim", "Simple");
  Tim->add_card(Card(Card::RANK_JACK, Card::SUIT_DIAMONDS));
  Tim->add_card(Card(Card::RANK_JACK, Card::SUIT_HEARTS));
  Tim->add_card(Card(Card::RANK_TEN, Card::SUIT_CLUBS));
  Tim->add_card(Card(Card::RANK_KING, Card::SUIT_HEARTS));
  Tim->add_card(Card(Card::RANK_ACE, Card::SUIT_CLUBS));


  // Bob makes tump
  Card nine_diamonds(Card::RANK_NINE, Card::SUIT_DIAMONDS);
  string trump;
  bool orderup = Tim->make_trump(
    nine_diamonds,    // Upcard
    true,           // Bob is also the dealer
    1,              // First round
    trump           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_TRUE(orderup);

  delete Tim;
}

TEST(test_simple_player_make_trump12) {
  Player * Tom = Player_factory("Tim", "Simple");
  Tom->add_card(Card(Card::RANK_JACK, Card::SUIT_DIAMONDS));
  Tom->add_card(Card(Card::RANK_JACK, Card::SUIT_HEARTS));
  Tom->add_card(Card(Card::RANK_TEN, Card::SUIT_CLUBS));
  Tom->add_card(Card(Card::RANK_KING, Card::SUIT_HEARTS));
  Tom->add_card(Card(Card::RANK_ACE, Card::SUIT_CLUBS));


  // Bob makes tump
  Card nine_diamonds(Card::RANK_NINE, Card::SUIT_DIAMONDS);
  string trump;
  bool orderup = Tom->make_trump(
    nine_diamonds,    // Upcard
    true,           // Bob is also the dealer
    1,              // First round
    trump           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_TRUE(orderup);

  delete Tom;
}

TEST(test_simple_player_make_trump13) {
  Player * Tim = Player_factory("Tim", "Simple");
  Tim->add_card(Card(Card::RANK_JACK, Card::SUIT_HEARTS));
  Tim->add_card(Card(Card::RANK_TEN, Card::SUIT_DIAMONDS));
  Tim->add_card(Card(Card::RANK_NINE, Card::SUIT_DIAMONDS));
  Tim->add_card(Card(Card::RANK_KING, Card::SUIT_HEARTS));
  Tim->add_card(Card(Card::RANK_ACE, Card::SUIT_CLUBS));


  // Bob makes tump
  Card ten_diamonds(Card::RANK_TEN, Card::SUIT_DIAMONDS);
  string trump;
  bool orderup = Tim->make_trump(
    ten_diamonds,    // Upcard
    true,           // Bob is also the dealer
    1,              // First round
    trump           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_FALSE(orderup);

  delete Tim;
}

TEST(test_simple_player_make_trump14) {
  Player * Tom = Player_factory("Tim", "Simple");
  Tom->add_card(Card(Card::RANK_JACK, Card::SUIT_HEARTS));
  Tom->add_card(Card(Card::RANK_TEN, Card::SUIT_DIAMONDS));
  Tom->add_card(Card(Card::RANK_NINE, Card::SUIT_DIAMONDS));
  Tom->add_card(Card(Card::RANK_KING, Card::SUIT_HEARTS));
  Tom->add_card(Card(Card::RANK_ACE, Card::SUIT_CLUBS));


  // Bob makes tump
  Card ten_diamonds(Card::RANK_TEN, Card::SUIT_DIAMONDS);
  string trump;
  bool orderup = Tom->make_trump(
    ten_diamonds,    // Upcard
    true,           // Bob is also the dealer
    1,              // First round
    trump           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_FALSE(orderup);

  delete Tom;
}

TEST(test_simple_player_make_trump15) {
  Player * Tim = Player_factory("Tim", "Simple");
  Tim->add_card(Card(Card::RANK_JACK, Card::SUIT_DIAMONDS));
  Tim->add_card(Card(Card::RANK_TEN, Card::SUIT_DIAMONDS));
  Tim->add_card(Card(Card::RANK_NINE, Card::SUIT_DIAMONDS));
  Tim->add_card(Card(Card::RANK_TEN, Card::SUIT_HEARTS));
  Tim->add_card(Card(Card::RANK_JACK, Card::SUIT_CLUBS));


  // Bob makes tump
  Card ten_diamonds(Card::RANK_TEN, Card::SUIT_DIAMONDS);
  string trump;
  bool orderup = Tim->make_trump(
    ten_diamonds,    // Upcard
    true,           // Bob is also the dealer
    1,              // First round
    trump           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_FALSE(orderup);

  delete Tim;
}

TEST(test_simple_player_make_trump16) {
  Player * Tom = Player_factory("Tim", "Simple");
  Tom->add_card(Card(Card::RANK_JACK, Card::SUIT_DIAMONDS));
  Tom->add_card(Card(Card::RANK_TEN, Card::SUIT_DIAMONDS));
  Tom->add_card(Card(Card::RANK_NINE, Card::SUIT_DIAMONDS));
  Tom->add_card(Card(Card::RANK_TEN, Card::SUIT_HEARTS));
  Tom->add_card(Card(Card::RANK_JACK, Card::SUIT_CLUBS));


  // Bob makes tump
  Card ten_diamonds(Card::RANK_TEN, Card::SUIT_DIAMONDS);
  string trump;
  bool orderup = Tom->make_trump(
    ten_diamonds,    // Upcard
    true,           // Bob is also the dealer
    1,              // First round
    trump           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_FALSE(orderup);

  delete Tom;
}

TEST(test_simple_player_make_trump17) {
  Player * Tim = Player_factory("Tim", "Simple");
  Tim->add_card(Card(Card::RANK_KING, Card::SUIT_DIAMONDS));
  Tim->add_card(Card(Card::RANK_TEN, Card::SUIT_DIAMONDS));
  Tim->add_card(Card(Card::RANK_NINE, Card::SUIT_DIAMONDS));
  Tim->add_card(Card(Card::RANK_NINE, Card::SUIT_HEARTS));
  Tim->add_card(Card(Card::RANK_JACK, Card::SUIT_CLUBS));


  // Bob makes tump
  Card jack_diamonds(Card::RANK_JACK, Card::SUIT_DIAMONDS);
  string trump;
  bool orderup = Tim->make_trump(
    jack_diamonds,    // Upcard
    true,           // Bob is also the dealer
    1,              // First round
    trump           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_FALSE(orderup);

  delete Tim;
}

TEST(test_simple_player_make_trump18) {
  Player * Tom = Player_factory("Tom", "Simple");
  Tom->add_card(Card(Card::RANK_KING, Card::SUIT_DIAMONDS));
  Tom->add_card(Card(Card::RANK_TEN, Card::SUIT_DIAMONDS));
  Tom->add_card(Card(Card::RANK_NINE, Card::SUIT_DIAMONDS));
  Tom->add_card(Card(Card::RANK_NINE, Card::SUIT_HEARTS));
  Tom->add_card(Card(Card::RANK_JACK, Card::SUIT_CLUBS));


  // Bob makes tump
  Card jack_diamonds(Card::RANK_JACK, Card::SUIT_DIAMONDS);
  string trump;
  bool orderup = Tom->make_trump(
    jack_diamonds,    // Upcard
    true,           // Bob is also the dealer
    1,              // First round
    trump           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_FALSE(orderup);

  delete Tom;
}

TEST(test_simple_player1_make_trump2) {
  Player * Tim = Player_factory("Tim", "Simple");
  Tim->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  Tim->add_card(Card(Card::RANK_TEN, Card::SUIT_SPADES));
  Tim->add_card(Card(Card::RANK_QUEEN, Card::SUIT_SPADES));
  Tim->add_card(Card(Card::RANK_KING, Card::SUIT_SPADES));
  Tim->add_card(Card(Card::RANK_ACE, Card::SUIT_SPADES));


  // Bob makes tump
  Card nine_hearts(Card::RANK_NINE, Card::SUIT_HEARTS);
  string trump;
  bool orderup = Tim->make_trump(
    nine_hearts,    // Upcard
    true,           // Bob is also the dealer
    1,              // First round
    trump           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_FALSE(orderup);

  delete Tim;
}

TEST(test_simple_player1_make_trump3) {
  Player * Timi = Player_factory("Tim", "Simple");
  Timi->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  Timi->add_card(Card(Card::RANK_TEN, Card::SUIT_SPADES));
  Timi->add_card(Card(Card::RANK_QUEEN, Card::SUIT_SPADES));
  Timi->add_card(Card(Card::RANK_KING, Card::SUIT_SPADES));
  Timi->add_card(Card(Card::RANK_ACE, Card::SUIT_SPADES));


  // Bob makes tump
  Card nine_hearts(Card::RANK_NINE, Card::SUIT_HEARTS);
  string trump;
  bool orderup = Timi->make_trump(
    nine_hearts,    // Upcard
    false,           // Bob is also the dealer
    1,              // First round
    trump           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_FALSE(orderup);

  delete Timi;
}

TEST(test_simple_player1_make_trump4) {
  Player * Rock = Player_factory("Tim", "Simple");
  Rock->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  Rock->add_card(Card(Card::RANK_TEN, Card::SUIT_HEARTS));
  Rock->add_card(Card(Card::RANK_QUEEN, Card::SUIT_SPADES));
  Rock->add_card(Card(Card::RANK_KING, Card::SUIT_SPADES));
  Rock->add_card(Card(Card::RANK_ACE, Card::SUIT_SPADES));


  // Bob makes tump
  Card jack_hearts(Card::RANK_JACK, Card::SUIT_HEARTS);
  string trump;
  bool orderup = Rock->make_trump(
    jack_hearts,    // Upcard
    true,           // Bob is also the dealer
    2,              // First round
    trump           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_TRUE(orderup);
  ASSERT_EQUAL(trump, "Diamonds");

  delete Rock;
}

TEST(test_simple_player1_make_trump5) {
  Player * Rockie = Player_factory("Tim", "Simple");
  Rockie->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  Rockie->add_card(Card(Card::RANK_TEN, Card::SUIT_HEARTS));
  Rockie->add_card(Card(Card::RANK_QUEEN, Card::SUIT_CLUBS));
  Rockie->add_card(Card(Card::RANK_KING, Card::SUIT_SPADES));
  Rockie->add_card(Card(Card::RANK_ACE, Card::SUIT_CLUBS));


  // Bob makes tump
  Card king_hearts(Card::RANK_KING, Card::SUIT_HEARTS);
  string trump;
  bool orderup = Rockie->make_trump(
    king_hearts,    // Upcard
    false,           // Bob is also the dealer
    2,              // First round
    trump           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_FALSE(orderup);

  delete Rockie;
}

TEST(test_simple_player1_make_trump6) {
  Player * Chris = Player_factory("Tim", "Simple");
  Chris->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  Chris->add_card(Card(Card::RANK_TEN, Card::SUIT_HEARTS));
  Chris->add_card(Card(Card::RANK_QUEEN, Card::SUIT_SPADES));
  Chris->add_card(Card(Card::RANK_KING, Card::SUIT_SPADES));
  Chris->add_card(Card(Card::RANK_ACE, Card::SUIT_SPADES));


  // Bob makes tump
  Card nine_clubs(Card::RANK_NINE, Card::SUIT_CLUBS);
  string trump;
  bool orderup = Chris->make_trump(
    nine_clubs,    // Upcard
    false,           // Bob is also the dealer
    2,              // First round
    trump           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_TRUE(orderup);
  ASSERT_EQUAL(trump, "Spades");

  delete Chris;
}

TEST(test_simple_player1_make_trump7) {
  Player * Christa = Player_factory("Tim", "Simple");
  Christa->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  Christa->add_card(Card(Card::RANK_TEN, Card::SUIT_HEARTS));
  Christa->add_card(Card(Card::RANK_QUEEN, Card::SUIT_SPADES));
  Christa->add_card(Card(Card::RANK_KING, Card::SUIT_SPADES));
  Christa->add_card(Card(Card::RANK_ACE, Card::SUIT_SPADES));


  // Bob makes tump
  Card jack_diamonds(Card::RANK_JACK, Card::SUIT_DIAMONDS);
  string trump1;
  bool orderup1 = Christa->make_trump(
    jack_diamonds,    // Upcard
    true,           // Bob is also the dealer
    2,              // First round
    trump1           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_TRUE(orderup1);
  ASSERT_EQUAL(trump1, "Hearts");

  delete Christa;

  Player * Belle = Player_factory("Tim", "Simple");
  Belle->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  Belle->add_card(Card(Card::RANK_KING, Card::SUIT_DIAMONDS));
  Belle->add_card(Card(Card::RANK_QUEEN, Card::SUIT_SPADES));
  Belle->add_card(Card(Card::RANK_KING, Card::SUIT_CLUBS));
  Belle->add_card(Card(Card::RANK_ACE, Card::SUIT_HEARTS));


  // Bob makes tump
  Card king_diamonds(Card::RANK_KING, Card::SUIT_DIAMONDS);
  string trump2;
  bool orderup2 = Belle->make_trump(
    king_diamonds,    // Upcard
    false,           // Bob is also the dealer
    2,              // First round
    trump2           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_TRUE(orderup2);
  ASSERT_EQUAL(trump2, "Hearts");
  delete Belle;

  Player * Belley = Player_factory("Tim", "Simple");
  Belley->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  Belley->add_card(Card(Card::RANK_KING, Card::SUIT_DIAMONDS));
  Belley->add_card(Card(Card::RANK_QUEEN, Card::SUIT_SPADES));
  Belley->add_card(Card(Card::RANK_KING, Card::SUIT_SPADES));
  Belley->add_card(Card(Card::RANK_ACE, Card::SUIT_SPADES));


  // Bob makes tump
  Card queen_spades(Card::RANK_QUEEN, Card::SUIT_SPADES);
  string trump3;
  bool orderup3 = Belley->make_trump(
    queen_spades,    // Upcard
    true,           // Bob is also the dealer
    2,              // First round
    trump3           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_TRUE(orderup3);
  ASSERT_EQUAL(trump3, "Clubs");
  delete Belley;

  Player * Rocko = Player_factory("Tim", "Simple");
  Rocko->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  Rocko->add_card(Card(Card::RANK_JACK, Card::SUIT_DIAMONDS));
  Rocko->add_card(Card(Card::RANK_QUEEN, Card::SUIT_SPADES));
  Rocko->add_card(Card(Card::RANK_KING, Card::SUIT_SPADES));
  Rocko->add_card(Card(Card::RANK_ACE, Card::SUIT_SPADES));


  // Bob makes tump
  Card ace_spades(Card::RANK_ACE, Card::SUIT_SPADES);
  string trump4;
  bool orderup4 = Rocko->make_trump(
    ace_spades,    // Upcard
    false,           // Bob is also the dealer
    2,              // First round
    trump4           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_FALSE(orderup4);

  delete Rocko;

  Player * Rocks = Player_factory("Tim", "Simple");
  Rocks->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  Rocks->add_card(Card(Card::RANK_JACK, Card::SUIT_DIAMONDS));
  Rocks->add_card(Card(Card::RANK_QUEEN, Card::SUIT_SPADES));
  Rocks->add_card(Card(Card::RANK_KING, Card::SUIT_SPADES));
  Rocks->add_card(Card(Card::RANK_ACE, Card::SUIT_SPADES));


  // Bob makes tump
  Card nine_hearts(Card::RANK_NINE, Card::SUIT_HEARTS);
  string trump5;
  bool orderup5 = Rocks->make_trump(
    nine_hearts,    // Upcard
    true,           // Bob is also the dealer
    2,              // First round
    trump5           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_TRUE(orderup5);
  ASSERT_EQUAL(trump5, "Diamonds");

  delete Rocks;

  Player * Rockf = Player_factory("Tim", "Simple");
  Rockf->add_card(Card(Card::RANK_NINE, Card::SUIT_CLUBS));
  Rockf->add_card(Card(Card::RANK_TEN, Card::SUIT_DIAMONDS));
  Rockf->add_card(Card(Card::RANK_QUEEN, Card::SUIT_SPADES));
  Rockf->add_card(Card(Card::RANK_KING, Card::SUIT_HEARTS));
  Rockf->add_card(Card(Card::RANK_ACE, Card::SUIT_SPADES));


  // Bob makes tump
  string trump6;
  bool orderup6 = Rockf->make_trump(
    nine_hearts,    // Upcard
    false,           // Bob is also the dealer
    2,              // First round
    trump6           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_FALSE(orderup6);

  delete Rockf;

  Player * Rockq = Player_factory("Tim", "Simple");
  Rockq->add_card(Card(Card::RANK_NINE, Card::SUIT_CLUBS));
  Rockq->add_card(Card(Card::RANK_TEN, Card::SUIT_DIAMONDS));
  Rockq->add_card(Card(Card::RANK_QUEEN, Card::SUIT_SPADES));
  Rockq->add_card(Card(Card::RANK_KING, Card::SUIT_CLUBS));
  Rockq->add_card(Card(Card::RANK_ACE, Card::SUIT_SPADES));


  // Bob makes tump
  string trump7;
  bool orderup7 = Rockq->make_trump(
    ace_spades,    // Upcard
    true,           // Bob is also the dealer
    2,              // First round
    trump7           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_TRUE(orderup7);
  ASSERT_EQUAL(trump7, "Clubs");

  delete Rockq;
}


TEST(test_simple_player1_make_trump8) {
  Player * Rocka = Player_factory("Tim", "Simple");
  Rocka->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  Rocka->add_card(Card(Card::RANK_JACK, Card::SUIT_DIAMONDS));
  Rocka->add_card(Card(Card::RANK_QUEEN, Card::SUIT_HEARTS));
  Rocka->add_card(Card(Card::RANK_KING, Card::SUIT_SPADES));
  Rocka->add_card(Card(Card::RANK_ACE, Card::SUIT_CLUBS));


  // Bob makes tump
  Card ten_diamonds(Card::RANK_TEN, Card::SUIT_DIAMONDS);
  string trump8;
  bool orderup8 = Rocka->make_trump(
    ten_diamonds,    // Upcard
    false,           // Bob is also the dealer
    2,              // First round
    trump8           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_TRUE(orderup8);
  ASSERT_EQUAL(trump8, "Hearts");

  delete Rocka;

  Player * Rockb = Player_factory("Tim", "Simple");
  Rockb->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  Rockb->add_card(Card(Card::RANK_JACK, Card::SUIT_DIAMONDS));
  Rockb->add_card(Card(Card::RANK_QUEEN, Card::SUIT_HEARTS));
  Rockb->add_card(Card(Card::RANK_KING, Card::SUIT_SPADES));
  Rockb->add_card(Card(Card::RANK_ACE, Card::SUIT_SPADES));


  // Bob makes tump
  Card jack_diamonds(Card::RANK_JACK, Card::SUIT_DIAMONDS);
  string trump9;
  bool orderup9 = Rockb->make_trump(
    jack_diamonds,    // Upcard
    true,           // Bob is also the dealer
    2,              // First round
    trump9           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_TRUE(orderup9);
  ASSERT_EQUAL(trump9, "Hearts");

  delete Rockb;

  Player * Rockc = Player_factory("Tim", "Simple");
  Rockc->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  Rockc->add_card(Card(Card::RANK_JACK, Card::SUIT_HEARTS));
  Rockc->add_card(Card(Card::RANK_QUEEN, Card::SUIT_HEARTS));
  Rockc->add_card(Card(Card::RANK_KING, Card::SUIT_DIAMONDS));
  Rockc->add_card(Card(Card::RANK_ACE, Card::SUIT_DIAMONDS));


  // Bob makes tump
  Card king_clubs(Card::RANK_KING, Card::SUIT_CLUBS);
  string trump10;
  bool orderup10 = Rockc->make_trump(
    king_clubs,    // Upcard
    false,           // Bob is also the dealer
    2,              // First round
    trump10           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_FALSE(orderup10);

  delete Rockc;

  Player * Rockd = Player_factory("Tim", "Simple");
  Rockd->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  Rockd->add_card(Card(Card::RANK_TEN, Card::SUIT_CLUBS));
  Rockd->add_card(Card(Card::RANK_QUEEN, Card::SUIT_SPADES));
  Rockd->add_card(Card(Card::RANK_KING, Card::SUIT_DIAMONDS));
  Rockd->add_card(Card(Card::RANK_ACE, Card::SUIT_DIAMONDS));


  // Bob makes tump
  Card queen_clubs(Card::RANK_QUEEN, Card::SUIT_CLUBS);
  string trump11;
  bool orderup11 = Rockd->make_trump(
    queen_clubs,    // Upcard
    true,           // Bob is also the dealer
    2,              // First round
    trump11           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_TRUE(orderup11);
  ASSERT_EQUAL(trump11, "Spades");

  delete Rockd;

  Player * Rocke = Player_factory("Tim", "Simple");
  Rocke->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  Rocke->add_card(Card(Card::RANK_TEN, Card::SUIT_DIAMONDS));
  Rocke->add_card(Card(Card::RANK_NINE, Card::SUIT_DIAMONDS));
  Rocke->add_card(Card(Card::RANK_KING, Card::SUIT_SPADES));
  Rocke->add_card(Card(Card::RANK_ACE, Card::SUIT_CLUBS));


  // Bob makes tump
  Card nine_hearts(Card::RANK_NINE, Card::SUIT_HEARTS);
  string trump12;
  bool orderup12 = Rocke->make_trump(
    nine_hearts,    // Upcard
    false,           // Bob is also the dealer
    2,              // First round
    trump12           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_FALSE(orderup12);

  delete Rocke;

  Player * Rockg = Player_factory("Tim", "Simple");
  Rockg->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  Rockg->add_card(Card(Card::RANK_TEN, Card::SUIT_DIAMONDS));
  Rockg->add_card(Card(Card::RANK_NINE, Card::SUIT_DIAMONDS));
  Rockg->add_card(Card(Card::RANK_KING, Card::SUIT_SPADES));
  Rockg->add_card(Card(Card::RANK_ACE, Card::SUIT_CLUBS));


  // Bob makes tump
  Card jack_hearts(Card::RANK_JACK, Card::SUIT_HEARTS);
  string trump13;
  bool orderup13 = Rockg->make_trump(
    jack_hearts,    // Upcard
    false,           // Bob is also the dealer
    2,              // First round
    trump13           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_FALSE(orderup13);

  delete Rockg;

  Player * Rockh = Player_factory("Tim", "Simple");
  Rockh->add_card(Card(Card::RANK_NINE, Card::SUIT_HEARTS));
  Rockh->add_card(Card(Card::RANK_ACE, Card::SUIT_HEARTS));
  Rockh->add_card(Card(Card::RANK_QUEEN, Card::SUIT_HEARTS));
  Rockh->add_card(Card(Card::RANK_KING, Card::SUIT_HEARTS));
  Rockh->add_card(Card(Card::RANK_ACE, Card::SUIT_HEARTS));


  // Bob makes tump
  string trump14;
  bool orderup14 = Rockh->make_trump(
    nine_hearts,    // Upcard
    false,           // Bob is also the dealer
    2,              // First round
    trump14           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_FALSE(orderup14);

  delete Rockh;

  Player * Rocki = Player_factory("Tim", "Simple");
  Rocki->add_card(Card(Card::RANK_NINE, Card::SUIT_HEARTS));
  Rocki->add_card(Card(Card::RANK_ACE, Card::SUIT_HEARTS));
  Rocki->add_card(Card(Card::RANK_QUEEN, Card::SUIT_HEARTS));
  Rocki->add_card(Card(Card::RANK_KING, Card::SUIT_HEARTS));
  Rocki->add_card(Card(Card::RANK_ACE, Card::SUIT_HEARTS));


  // Bob makes tump
  string trump15;
  bool orderup15 = Rocki->make_trump(
    nine_hearts,    // Upcard
    true,           // Bob is also the dealer
    2,              // First round
    trump15           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_TRUE(orderup15);
  ASSERT_EQUAL(trump15, "Diamonds");

  delete Rocki;

  Player * Rockj = Player_factory("Tim", "Simple");
  Rockj->add_card(Card(Card::RANK_NINE, Card::SUIT_DIAMONDS));
  Rockj->add_card(Card(Card::RANK_TEN, Card::SUIT_DIAMONDS));
  Rockj->add_card(Card(Card::RANK_QUEEN, Card::SUIT_HEARTS));
  Rockj->add_card(Card(Card::RANK_KING, Card::SUIT_HEARTS));
  Rockj->add_card(Card(Card::RANK_ACE, Card::SUIT_HEARTS));


  // Bob makes tump
  string trump16;
  bool orderup16 = Rockj->make_trump(
    jack_hearts,    // Upcard
    false,           // Bob is also the dealer
    2,              // First round
    trump16           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_FALSE(orderup16);
  delete Rockj;

  Player * Rockk = Player_factory("Tim", "Simple");
  Rockk->add_card(Card(Card::RANK_NINE, Card::SUIT_DIAMONDS));
  Rockk->add_card(Card(Card::RANK_TEN, Card::SUIT_DIAMONDS));
  Rockk->add_card(Card(Card::RANK_QUEEN, Card::SUIT_HEARTS));
  Rockk->add_card(Card(Card::RANK_KING, Card::SUIT_HEARTS));
  Rockk->add_card(Card(Card::RANK_ACE, Card::SUIT_HEARTS));


  // Bob makes tump
  Card jack_clubs(Card::RANK_JACK, Card::SUIT_CLUBS);
  string trump17;
  bool orderup17 = Rockk->make_trump(
    jack_clubs,    // Upcard
    true,           // Bob is also the dealer
    2,              // First round
    trump17           // Suit ordered up (if any)
  );

  // Verify Bob's order up and trump suit
  ASSERT_TRUE(orderup17);
  ASSERT_EQUAL(trump17, "Spades");
  delete Rockk;
}

TEST(test_simple_player_lead_card) {
  // Bob's hand
  Player * bob = Player_factory("Bob", "Simple");
  bob->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_TEN, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_QUEEN, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_KING, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_ACE, Card::SUIT_SPADES));

  // Bob adds a card to his hand and discards one card
  bob->add_and_discard(
    Card(Card::RANK_NINE, Card::SUIT_HEARTS) // upcard
  );
    

  // Bob leads
  Card card_led = bob->lead_card(Card::SUIT_HEARTS);

  // Verify the card Bob selected to lead
  Card ace_spades(Card::RANK_ACE, Card::SUIT_SPADES);
  ASSERT_EQUAL(card_led, ace_spades); //check led card

  delete bob;
}

TEST(test_simple_player_lead_card1) {
  // Bob's hand
  Player * bob = Player_factory("Bob", "Simple");
  bob->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_TEN, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_QUEEN, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_ACE, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_KING, Card::SUIT_SPADES));

  // Bob adds a card to his hand and discards one card
  bob->add_and_discard(
    Card(Card::RANK_JACK, Card::SUIT_SPADES) // upcard
  );
    

  // Bob leads
  Card card_led = bob->lead_card(Card::SUIT_SPADES);

  // Verify the card Bob selected to lead
  Card jack_spades(Card::RANK_JACK, Card::SUIT_SPADES);
  ASSERT_EQUAL(card_led, jack_spades); //check led card

  delete bob;
}

TEST(test_simple_player_lead_card2) {
  // Bob's hand
  Player * bob = Player_factory("Bob", "Simple");
  bob->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_TEN, Card::SUIT_CLUBS));
  bob->add_card(Card(Card::RANK_JACK, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_ACE, Card::SUIT_DIAMONDS));
  bob->add_card(Card(Card::RANK_KING, Card::SUIT_SPADES));

  // Bob adds a card to his hand and discards one card
  bob->add_and_discard(
    Card(Card::RANK_QUEEN, Card::SUIT_DIAMONDS) // upcard
  );
    

  // Bob leads
  Card card_led = bob->lead_card(Card::SUIT_DIAMONDS);

  // Verify the card Bob selected to lead
  Card king_spades(Card::RANK_KING, Card::SUIT_SPADES);
  ASSERT_EQUAL(card_led, king_spades); //check led card

  delete bob;
}

TEST(test_simple_player_lead_card3) {
  // Bob's hand
  Player * bob = Player_factory("Bob", "Simple");
  bob->add_card(Card(Card::RANK_NINE, Card::SUIT_CLUBS));
  bob->add_card(Card(Card::RANK_TEN, Card::SUIT_CLUBS));
  bob->add_card(Card(Card::RANK_JACK, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_ACE, Card::SUIT_CLUBS));
  bob->add_card(Card(Card::RANK_KING, Card::SUIT_SPADES));

  // Bob adds a card to his hand and discards one card
  bob->add_and_discard(
    Card(Card::RANK_QUEEN, Card::SUIT_CLUBS) // upcard
  );
    

  // Bob leads
  Card card_led = bob->lead_card(Card::SUIT_CLUBS);

  // Verify the card Bob selected to lead
  Card jack_spades(Card::RANK_JACK, Card::SUIT_SPADES);
  ASSERT_EQUAL(card_led, jack_spades); //check led card

  delete bob;
}

TEST(test_simple_player_lead_card4) {
  // Bob's hand
  Player * bob = Player_factory("Bob", "Simple");
  bob->add_card(Card(Card::RANK_NINE, Card::SUIT_CLUBS));
  bob->add_card(Card(Card::RANK_TEN, Card::SUIT_CLUBS));
  bob->add_card(Card(Card::RANK_JACK, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_ACE, Card::SUIT_CLUBS));
  bob->add_card(Card(Card::RANK_KING, Card::SUIT_CLUBS));

  // Bob adds a card to his hand and discards one card
  bob->add_and_discard(
    Card(Card::RANK_QUEEN, Card::SUIT_CLUBS) // upcard
  );
    

  // Bob leads
  Card card_led = bob->lead_card(Card::SUIT_CLUBS);

  // Verify the card Bob selected to lead
  Card jack_spades(Card::RANK_JACK, Card::SUIT_SPADES);
  ASSERT_EQUAL(card_led, jack_spades); //check led card

  delete bob;
}

TEST(test_simple_player_lead_card5) {
  // Bob's hand
  Player * bob = Player_factory("Bob", "Simple");
  bob->add_card(Card(Card::RANK_NINE, Card::SUIT_CLUBS));
  bob->add_card(Card(Card::RANK_TEN, Card::SUIT_CLUBS));
  bob->add_card(Card(Card::RANK_JACK, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_TEN, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_KING, Card::SUIT_CLUBS));

  // Bob adds a card to his hand and discards one card
  bob->add_and_discard(
    Card(Card::RANK_JACK, Card::SUIT_CLUBS) // upcard
  );
    

  // Bob leads
  Card card_led = bob->lead_card(Card::SUIT_CLUBS);

  // Verify the card Bob selected to lead
  Card jack_spades(Card::RANK_JACK, Card::SUIT_CLUBS);
  ASSERT_EQUAL(card_led, jack_spades); //check led card

  delete bob;
}

TEST(test_simple_player_lead_card6) {
  // Bob's hand
  Player * bob = Player_factory("Bob", "Simple");
  bob->add_card(Card(Card::RANK_NINE, Card::SUIT_CLUBS));
  bob->add_card(Card(Card::RANK_TEN, Card::SUIT_DIAMONDS));
  bob->add_card(Card(Card::RANK_JACK, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_ACE, Card::SUIT_HEARTS));
  bob->add_card(Card(Card::RANK_KING, Card::SUIT_CLUBS));

  // Bob adds a card to his hand and discards one card
  bob->add_and_discard(
    Card(Card::RANK_NINE, Card::SUIT_CLUBS) // upcard
  );
    

  // Bob leads
  Card card_led = bob->lead_card(Card::SUIT_CLUBS);

  // Verify the card Bob selected to lead
  Card ace_hearts(Card::RANK_ACE, Card::SUIT_HEARTS);
  ASSERT_EQUAL(card_led, ace_hearts); //check led card

  delete bob;
}

TEST(test_simple_player_lead_card7) {
  // Bob's hand
  Player * bob = Player_factory("Bob", "Simple");
  bob->add_card(Card(Card::RANK_NINE, Card::SUIT_CLUBS));
  bob->add_card(Card(Card::RANK_QUEEN, Card::SUIT_HEARTS));
  bob->add_card(Card(Card::RANK_JACK, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_ACE, Card::SUIT_HEARTS));
  bob->add_card(Card(Card::RANK_KING, Card::SUIT_SPADES));

  // Bob adds a card to his hand and discards one card
  bob->add_and_discard(
    Card(Card::RANK_TEN, Card::SUIT_DIAMONDS) // upcard
  );
    

  // Bob leads
  Card card_led = bob->lead_card(Card::SUIT_DIAMONDS);

  // Verify the card Bob selected to lead
  Card ace_hearts(Card::RANK_ACE, Card::SUIT_HEARTS);
  ASSERT_EQUAL(card_led, ace_hearts); //check led card

  delete bob;
}

TEST(test_simple_player_play_card) {
  // Bob's hand
  Player * bob = Player_factory("Bob", "Simple");
  bob->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_TEN, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_QUEEN, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_KING, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_ACE, Card::SUIT_SPADES));

  // Bob plays a card
  Card nine_diamonds(Card::RANK_NINE, Card::SUIT_DIAMONDS);
  Card card_played = bob->play_card(
    nine_diamonds,  // Nine of Diamonds is led
    "Hearts"        // Trump suit
  );

  // Verify the card Bob played
  ASSERT_EQUAL(card_played, Card(Card::RANK_NINE, Card::SUIT_SPADES));
  delete bob;
}

TEST(test_simple_player_play_card2) {
  // Bob's hand
  Player * bob = Player_factory("Bob", "Simple");
  bob->add_card(Card(Card::RANK_NINE, Card::SUIT_HEARTS));
  bob->add_card(Card(Card::RANK_TEN, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_QUEEN, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_KING, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_ACE, Card::SUIT_SPADES));

  // Bob plays a card
  Card nine_spades(Card::RANK_NINE, Card::SUIT_SPADES);
  Card card_played = bob->play_card(
    nine_spades,  // Nine of Diamonds is led
    "Hearts"        // Trump suit
  );

  // Verify the card Bob played
  ASSERT_EQUAL(card_played, Card(Card::RANK_ACE, Card::SUIT_SPADES));
  delete bob;
}

TEST(test_simple_player_play_card3) {
  // Bob's hand
  Player * bob = Player_factory("Bob", "Simple");
  bob->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_TEN, Card::SUIT_CLUBS));
  bob->add_card(Card(Card::RANK_QUEEN, Card::SUIT_HEARTS));
  bob->add_card(Card(Card::RANK_KING, Card::SUIT_DIAMONDS));
  bob->add_card(Card(Card::RANK_ACE, Card::SUIT_SPADES));

  // Bob plays a card
  Card jack_hearts(Card::RANK_JACK, Card::SUIT_HEARTS);
  Card card_played = bob->play_card(
    jack_hearts,  // Nine of Diamonds is led
    "Hearts"        // Trump suit
  );

  // Verify the card Bob played
  ASSERT_EQUAL(card_played, Card(Card::RANK_QUEEN, Card::SUIT_HEARTS));
  delete bob;
}

TEST(test_simple_player_play_card4) {
  // Bob's hand
  Player * bob = Player_factory("Bob", "Simple");
  bob->add_card(Card(Card::RANK_JACK, Card::SUIT_HEARTS));
  bob->add_card(Card(Card::RANK_TEN, Card::SUIT_HEARTS));
  bob->add_card(Card(Card::RANK_QUEEN, Card::SUIT_HEARTS));
  bob->add_card(Card(Card::RANK_KING, Card::SUIT_HEARTS));
  bob->add_card(Card(Card::RANK_ACE, Card::SUIT_HEARTS));

  // Bob plays a card
  Card nine_diamonds(Card::RANK_NINE, Card::SUIT_DIAMONDS);
  Card card_played = bob->play_card(
    nine_diamonds,  // Nine of Diamonds is led
    "Hearts"        // Trump suit
  );

  // Verify the card Bob played
  ASSERT_EQUAL(card_played, Card(Card::RANK_TEN, Card::SUIT_HEARTS));
  delete bob;
}

TEST(test_simple_player_play_card5) {
  // Bob's hand
  Player * bob = Player_factory("Bob", "Simple");
  bob->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_TEN, Card::SUIT_CLUBS));
  bob->add_card(Card(Card::RANK_JACK, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_KING, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_ACE, Card::SUIT_SPADES));

  // Bob plays a card
  Card ten_diamonds(Card::RANK_TEN, Card::SUIT_DIAMONDS);
  Card card_played = bob->play_card(
    ten_diamonds,  // Nine of Diamonds is led
    "Spades"        // Trump suit
  );

  // Verify the card Bob played
  ASSERT_EQUAL(card_played, Card(Card::RANK_TEN, Card::SUIT_CLUBS));
  delete bob;
}

TEST(test_simple_player_play_card6) {
  // Bob's hand
  Player * bob = Player_factory("Bob", "Simple");
  bob->add_card(Card(Card::RANK_JACK, Card::SUIT_DIAMONDS));
  bob->add_card(Card(Card::RANK_ACE, Card::SUIT_DIAMONDS));
  // Bob plays a card
  Card nine_diamonds(Card::RANK_NINE, Card::SUIT_DIAMONDS);
  Card card_played = bob->play_card(
    nine_diamonds,  // Nine of Diamonds is led
    "Diamonds"        // Trump suit
  );

  // Verify the card Bob played
  ASSERT_EQUAL(card_played, Card(Card::RANK_JACK, Card::SUIT_DIAMONDS));
  delete bob;
}

TEST(test_simple_player_play_card7) {
  // Bob's hand
  Player * bob = Player_factory("Bob", "Simple");
  bob->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_JACK, Card::SUIT_HEARTS));
  bob->add_card(Card(Card::RANK_JACK, Card::SUIT_CLUBS));
  bob->add_card(Card(Card::RANK_JACK, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_JACK, Card::SUIT_DIAMONDS));

  // Bob plays a card
  Card queen_clubs(Card::RANK_QUEEN, Card::SUIT_CLUBS);
  Card card_played = bob->play_card(
    queen_clubs,  // Nine of Diamonds is led
    "Spades"        // Trump suit
  );

  // Verify the card Bob played
  ASSERT_EQUAL(card_played, Card(Card::RANK_JACK, Card::SUIT_HEARTS));
  delete bob;
}

TEST(test_simple_player_play_card8) {
  // Bob's hand
  Player * bob = Player_factory("Bob", "Simple");
  bob->add_card(Card(Card::RANK_JACK, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_TEN, Card::SUIT_HEARTS));
  bob->add_card(Card(Card::RANK_QUEEN, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_KING, Card::SUIT_DIAMONDS));
  bob->add_card(Card(Card::RANK_ACE, Card::SUIT_HEARTS));

  // Bob plays a card
  Card queen_clubs(Card::RANK_QUEEN, Card::SUIT_CLUBS);
  Card card_played = bob->play_card(
    queen_clubs,  // Nine of Diamonds is led
    "Clubs"        // Trump suit
  );

  // Verify the card Bob played
  ASSERT_EQUAL(card_played, Card(Card::RANK_JACK, Card::SUIT_SPADES));
  delete bob;
}

TEST(test_simple_player_play_card9) {
  // Bob's hand
  Player * bob = Player_factory("Bob", "Simple");
  bob->add_card(Card(Card::RANK_JACK, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_TEN, Card::SUIT_HEARTS));
  bob->add_card(Card(Card::RANK_QUEEN, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_KING, Card::SUIT_DIAMONDS));
  bob->add_card(Card(Card::RANK_ACE, Card::SUIT_HEARTS));

  // Bob plays a card
  Card ace_spades(Card::RANK_ACE, Card::SUIT_SPADES);
  Card card_played = bob->play_card(
    ace_spades,  // Nine of Diamonds is led
    "Spades"        // Trump suit
  );

  // Verify the card Bob played
  ASSERT_EQUAL(card_played, Card(Card::RANK_JACK, Card::SUIT_SPADES));
  delete bob;
}

TEST(test_simple_player_play_card10) {//FIXME
  // Bob's hand
  Player * bob = Player_factory("Bob", "Simple");
  bob->add_card(Card(Card::RANK_JACK, Card::SUIT_CLUBS));
  bob->add_card(Card(Card::RANK_TEN, Card::SUIT_HEARTS));
  bob->add_card(Card(Card::RANK_QUEEN, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_KING, Card::SUIT_DIAMONDS));
  bob->add_card(Card(Card::RANK_ACE, Card::SUIT_HEARTS));

  // Bob plays a card
  Card ace_spades(Card::RANK_ACE, Card::SUIT_SPADES);
  Card card_played = bob->play_card(
    ace_spades,  // Nine of Diamonds is led
    "Spades"        // Trump suit
  );

  // Verify the card Bob played
  ASSERT_EQUAL(card_played, Card(Card::RANK_JACK, Card::SUIT_CLUBS));
  delete bob;
}

TEST(test_simple_player_play_card11) {
  // Bob's hand
  Player * bob = Player_factory("Bob", "Simple");
  bob->add_card(Card(Card::RANK_JACK, Card::SUIT_CLUBS));
  bob->add_card(Card(Card::RANK_TEN, Card::SUIT_HEARTS));
  bob->add_card(Card(Card::RANK_JACK, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_KING, Card::SUIT_DIAMONDS));
  bob->add_card(Card(Card::RANK_ACE, Card::SUIT_HEARTS));

  // Bob plays a card
  Card ace_clubs(Card::RANK_ACE, Card::SUIT_CLUBS);
  Card card_played = bob->play_card(
    ace_clubs,  // Nine of Diamonds is led
    "Spades"        // Trump suit
  );

  // Verify the card Bob played
  ASSERT_EQUAL(card_played, Card(Card::RANK_TEN, Card::SUIT_HEARTS));
  delete bob;
}

TEST(test_simple_player_play_card12) {
  // Bob's hand
  Player * bob = Player_factory("Bob", "Simple");
  bob->add_card(Card(Card::RANK_NINE, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_JACK, Card::SUIT_HEARTS));
  bob->add_card(Card(Card::RANK_JACK, Card::SUIT_CLUBS));
  bob->add_card(Card(Card::RANK_JACK, Card::SUIT_SPADES));
  bob->add_card(Card(Card::RANK_KING, Card::SUIT_CLUBS));

  // Bob plays a card
  Card queen_clubs(Card::RANK_QUEEN, Card::SUIT_CLUBS);
  Card card_played = bob->play_card(
    queen_clubs,  // Nine of Diamonds is led
    "Spades"        // Trump suit
  );

  // Verify the card Bob played
  ASSERT_EQUAL(card_played, Card(Card::RANK_KING, Card::SUIT_CLUBS));
  delete bob;
}

TEST_MAIN()
