procedure Test is

   X : Integer := 42;

   procedure F(E : out Integer) is
   begin
      E := 1;
   end F;

   procedure G(E : out Integer) is
   begin
      E := X;
   end G;

   P : access procedure (E : out Integer);
   B : Boolean;
   R : Integer;
begin
   if B then
      P := F'Access;
   else
      P := G'Access;
   end if;
   P(R);
end Test;
