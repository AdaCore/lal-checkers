procedure A is
   C_A : String := "abc";
   Char : Character;

   function X (A : String ; I : Integer) return Character is
   begin
      return A (I);
   end X;
begin
   Char := X (C_A, 2);
end A;
