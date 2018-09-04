procedure Test is
   procedure F(R : out Integer; X : Integer := 0; Z : Integer) is
   begin
      null;
   end F;

   R : Integer;
begin
   F(R, Z => 3);
end Ex1;
