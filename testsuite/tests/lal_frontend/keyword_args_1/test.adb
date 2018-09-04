procedure Test is
   function F(X, Y, Z : Integer) return Integer is
   begin
      return 0;
   end F;

   R : Integer := F(2, Z => 4, Y => 3);
begin
   null;
end Ex1;
