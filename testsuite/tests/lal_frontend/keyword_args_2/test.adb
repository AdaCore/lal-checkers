procedure Test is
   function F(X, Y : Integer; Z : out Integer) return Integer is
   begin
      Z := 3;
      return 0;
   end F;

   A : Integer;
   R : Integer := F(Z => A, X => 1, Y => 13);
begin
   null;
end Ex1;
